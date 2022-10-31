#ifdef __clang__
# pragma clang diagnostic push
# pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
# pragma clang diagnostic ignored "-Wc++1z-extensions"
#elif defined __GNUC__
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wpedantic"
# pragma GCC diagnostic ignored "-Wunused-function"
#endif

#include "lamb.h"

#include <torch/csrc/autograd/variable.h>
#include <torch/nn/module.h>
#include <torch/serialize/archive.h>
#include <torch/utils.h>

#include <ATen/ATen.h>
#include <c10/util/irange.h>

#include <cmath>
#include <functional>

namespace torch {
namespace optim {

LambOptions::LambOptions(double lr) : lr_(lr) {}

bool operator==(const LambOptions& lhs, const LambOptions& rhs) {
  return (lhs.lr() == rhs.lr()) &&
         (std::get<0>(lhs.betas()) == std::get<0>(rhs.betas())) &&
         (std::get<1>(lhs.betas()) == std::get<1>(rhs.betas())) &&
         (lhs.eps() == rhs.eps()) &&
         (lhs.weight_decay() == rhs.weight_decay() &&
         (lhs.unbiased() == rhs.unbiased()));
}

void LambOptions::serialize(torch::serialize::OutputArchive& archive) const {
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(lr);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(betas);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(eps);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(weight_decay);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(unbiased);
}

void LambOptions::serialize(torch::serialize::InputArchive& archive) {
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, lr);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(betas_t, betas);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, eps);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, weight_decay);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(bool, unbiased);
}

double LambOptions::get_lr() const {
  return lr();
}

void LambOptions::set_lr(const double lr) {
  this->lr(lr);
}

bool operator==(const LambParamState& lhs, const LambParamState& rhs) {
  return (lhs.step() == rhs.step()) &&
          torch::equal(lhs.exp_avg(), rhs.exp_avg()) &&
          torch::equal(lhs.exp_avg_sq(), rhs.exp_avg_sq());
}

void LambParamState::serialize(torch::serialize::OutputArchive& archive) const {
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(step);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(exp_avg);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(exp_avg_sq);
}

void LambParamState::serialize(torch::serialize::InputArchive& archive) {
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(int64_t, step);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(Tensor, exp_avg);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(Tensor, exp_avg_sq);
}

Tensor Lamb::step(LossClosure closure)  {
  NoGradGuard no_grad;
  Tensor loss = {};
  if (closure != nullptr) {
    torch::AutoGradMode enable_grad(true);
    loss = closure();
  }

  // calculate global gradient norm if flag set true
  Tensor gnorm;
  if(static_cast<LambOptions&>(defaults()).globalnorm()) {
    for (auto& group : param_groups_) {
      for (auto& p : group.params()) {
        if (!p.grad().defined()) {
          continue;
        }
        TORCH_CHECK(!p.grad().is_sparse(), "lamb: sparse gradients not implemented");
        if(!gnorm.defined())
         gnorm=torch::zeros(1,TensorOptions().device(p.device()).dtype(p.dtype()));
        gnorm.add_(p.grad().pow(2).sum());
      }
    }
    gnorm.sqrt_();
  }

  for (auto& group : param_groups_) {
    for (auto& p : group.params()) {
      if (!p.grad().defined()) {
        continue;
      }
      auto grad = p.grad();
      TORCH_CHECK(!grad.is_sparse(), "lamb: sparse gradients not implemented");
      auto param_state = state_.find(c10::guts::to_string(p.unsafeGetTensorImpl()));
      auto& options = static_cast<LambOptions&>(group.options());

      // State initialization
      if(param_state == state_.end()) {
        auto state = std::make_unique<LambParamState>();
        state->step(0);
        state->exp_avg(torch::zeros_like(p, MemoryFormat::Preserve));
        state->exp_avg_sq(torch::zeros_like(p, MemoryFormat::Preserve));
        state_[c10::guts::to_string(p.unsafeGetTensorImpl())] = std::move(state);
      }

      auto& state = static_cast<LambParamState&>(*state_[c10::guts::to_string(p.unsafeGetTensorImpl())]);
      auto& exp_avg = state.exp_avg();
      auto& exp_avg_sq = state.exp_avg_sq();

      state.step(state.step()+1);
      auto b1 = std::get<0>(options.betas());
      auto b2 = std::get<1>(options.betas());

      if(gnorm.defined())  // divide gradient by global gradient norm if defined
       grad.div_(gnorm);

      // Decay the first and second moment running average coefficient
      exp_avg.mul_(b1).add_(grad, 1 - b1);
      exp_avg_sq.mul_(b2).addcmul_(grad, grad, 1 - b2);

      auto upd = options.unbiased()
               ?      (exp_avg           /     (1-std::pow(b1,state.step())))
                    / (exp_avg_sq.sqrt() / sqrt(1-std::pow(b2,state.step()))).add_(options.eps())
               :  exp_avg / exp_avg_sq.sqrt().add_(options.eps());

      if(options.weight_decay() != 0) {
        upd.add_(p, options.weight_decay());
      }

      auto w_norm = p.norm(2.0);
      auto g_norm = upd.norm(2.0);
      auto trust_ratio = torch::where(torch::logical_and(w_norm>0,g_norm>0), w_norm/g_norm, torch::ones_like(w_norm));
      if(options.trustclip())
       trust_ratio.clip_(options.trustmin(),options.trustmax());
      p.add_(upd.mul_(trust_ratio), -options.lr());
    }
  }
  return loss;
}

void Lamb::save(serialize::OutputArchive& archive) const {
  serialize(*this, archive);
}

void Lamb::load(serialize::InputArchive& archive) {
  serialize(*this, archive);
}

} // namespace optim
} // namespace torch

#ifdef __clang__
# pragma clang diagnostic pop
#elif defined __GNUC__
# pragma GCC diagnostic pop
#endif
