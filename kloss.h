#pragma once

namespace {
  static inline at::Tensor apply_loss_reduction(const at::Tensor& unreduced, int64_t reduction) {
    if (reduction == at::Reduction::Mean) {
      return unreduced.mean();
    } else if (reduction == at::Reduction::Sum) {
      return unreduced.sum();
    }
    return unreduced;
  }
}

// ------------------------------------------------------------------------------------------
// redefine binary cross entropy loss without batch weights as part of initialization options
// move batch weight to optional 3rd argument of forward call
// ------------------------------------------------------------------------------------------
struct TORCH_API BCELossOptions {
  typedef c10::variant<torch::enumtype::kNone, torch::enumtype::kMean, torch::enumtype::kSum> reduction_t;
  TORCH_ARG(reduction_t, reduction) = torch::kMean;
};

struct TORCH_API BCEWithLogitsLossOptions {
  typedef c10::variant<torch::enumtype::kNone, torch::enumtype::kMean, torch::enumtype::kSum> reduction_t;
  TORCH_ARG(Tensor, pos_weight) = {};
  TORCH_ARG(reduction_t, reduction) = torch::kMean;
};

struct TORCH_API BCELossImpl : torch::nn::Cloneable<BCELossImpl> {
  explicit BCELossImpl(const BCELossOptions& options_ = {}) : options(options_) {reset();}
  void reset() override {}
  void pretty_print(std::ostream& stream) const override {stream << "BCELoss()";}
  Tensor forward(const Tensor& input, const Tensor& target, const Tensor& weight={}) {
   return torch::nn::functional::detail::binary_cross_entropy(input, target, weight, options.reduction());
  }
  BCELossOptions options;
};

struct TORCH_API BCEWithLogitsLossImpl : public torch::nn::Cloneable<BCEWithLogitsLossImpl> {
  explicit BCEWithLogitsLossImpl(const BCEWithLogitsLossOptions& options_ = {}) : options(options_) {reset();}
  void reset() override {pos_weight=register_buffer("pos_weight", options.pos_weight());}
  void pretty_print(std::ostream& stream) const override {stream << "BCEWithLogitsLoss()";}
  Tensor forward(const Tensor& input, const Tensor& target, const Tensor& weight={}) {
   return torch::nn::functional::detail::binary_cross_entropy_with_logits(input, target, weight, options.reduction(), options.pos_weight());
  }
  BCEWithLogitsLossOptions options;
  Tensor pos_weight;
};

TORCH_MODULE(BCELoss);
TORCH_MODULE(BCEWithLogitsLoss);

struct TORCH_API SmoothCrossEntropyOptions {
  typedef c10::variant<torch::enumtype::kNone, torch::enumtype::kMean, torch::enumtype::kSum> reduction_t;
  TORCH_OPTIONS_CTOR_VARIANT_ARG3(SmoothCrossEntropyOptions, reduction, kNone, kMean, kSum)
  TORCH_ARG(double, smoothing) = 0.1;
  TORCH_ARG(reduction_t, reduction) = torch::kMean;
};

struct TORCH_API SmoothCrossEntropyImpl : public torch::nn::Cloneable<SmoothCrossEntropyImpl> {
  explicit SmoothCrossEntropyImpl(const SmoothCrossEntropyOptions& options_ = {}) : options(options_) {reset();}
  void reset() override {}
  void pretty_print(std::ostream& stream) const override {stream << "SmoothCrossEntropy(smoothing=" << options.smoothing() << ")";}
 
  Tensor forward(const Tensor& input,const Tensor& target) {
   TORCH_CHECK(!target.is_floating_point(), "sce: smooth crossentropy expects integer/long target, given ",target.dtype());
   Tensor p = input.log_softmax(-1).neg();
   Tensor n = p.gather(-1,target.to(torch::kLong).unsqueeze(1)).squeeze(1);
   return apply_loss_reduction((1-options.smoothing()) * n + options.smoothing() * p.mean(-1), 
                               torch::enumtype::reduction_get_enum(options.reduction()));
  }
  SmoothCrossEntropyOptions options;
};
TORCH_MODULE(SmoothCrossEntropy);
