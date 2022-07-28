#pragma once

#include <torch/nn/module.h>
#include <torch/optim/optimizer.h>
#include <torch/optim/serialize.h>

#include <utility>
#include <vector>

namespace torch {
namespace serialize {
class OutputArchive;
class InputArchive;
} // namespace serialize
} // namespace torch

namespace torch {
namespace optim {

struct TORCH_API LambOptions : public OptimizerCloneableOptions<LambOptions> {
  LambOptions(double lr = 1e-3);
  TORCH_ARG(double, lr) = 1e-3;
  typedef std::tuple<double, double> betas_t;
  TORCH_ARG(betas_t, betas) = std::make_tuple(0.9, 0.999);
  TORCH_ARG(double, eps) = 1e-8;
  TORCH_ARG(double, weight_decay) = 0;
  TORCH_ARG(bool, unbiased) = true;
  TORCH_ARG(bool, globalnorm) = true;
  TORCH_ARG(bool, trustclip) = true;
  TORCH_ARG(double, trustmin) = 0.0;
  TORCH_ARG(double, trustmax) = 1.0;
public:
  void serialize(torch::serialize::InputArchive& archive) override;
  void serialize(torch::serialize::OutputArchive& archive) const override;
  TORCH_API friend bool operator==(const LambOptions& lhs, const LambOptions& rhs);
  ~LambOptions() override = default;
  double get_lr() const override;
  void set_lr(const double lr) override;
};

struct TORCH_API LambParamState : public OptimizerCloneableParamState<LambParamState> {
  TORCH_ARG(int64_t, step) = 0;
  TORCH_ARG(torch::Tensor, exp_avg);
  TORCH_ARG(torch::Tensor, exp_avg_sq);
  TORCH_ARG(torch::Tensor, max_exp_avg_sq) = {};

public:
  void serialize(torch::serialize::InputArchive& archive) override;
  void serialize(torch::serialize::OutputArchive& archive) const override;
  TORCH_API friend bool operator==(const LambParamState& lhs, const LambParamState& rhs);
  ~LambParamState() override = default;
};

class TORCH_API Lamb : public Optimizer {
 public:
   explicit Lamb(std::vector<OptimizerParamGroup> param_groups,
       LambOptions defaults = {}) : Optimizer(std::move(param_groups), std::make_unique<LambOptions>(defaults)) {
     TORCH_CHECK(defaults.lr() >= 0, "Invalid learning rate: ", defaults.lr());
     TORCH_CHECK(defaults.eps() >= 0, "Invalid epsilon value: ", defaults.eps());
     auto betas = defaults.betas();
     TORCH_CHECK(0 <= std::get<0>(betas) && std::get<0>(betas) < 1.0, "Invalid beta parameter at index 0: ", std::get<0>(betas));
     TORCH_CHECK(0 <= std::get<1>(betas) && std::get<1>(betas) < 1.0, "Invalid beta parameter at index 1: ", std::get<1>(betas));
     TORCH_CHECK(defaults.weight_decay() >= 0, "Invalid weight_decay value: ", defaults.weight_decay());
   }
   explicit Lamb(
       std::vector<Tensor> params,
       // NOLINTNEXTLINE(performance-move-const-arg)
       LambOptions defaults = {}) : Lamb({std::move(OptimizerParamGroup(params))}, defaults) {}

  torch::Tensor step(LossClosure closure = nullptr) override;
  void save(serialize::OutputArchive& archive) const override;
  void load(serialize::InputArchive& archive) override;

 private:
  template <typename Self, typename Archive>
  static void serialize(Self& self, Archive& archive) {
    _TORCH_OPTIM_SERIALIZE_WITH_TEMPLATE_ARG(Lamb);
  }
};

} // namespace optim
} // namespace torch
