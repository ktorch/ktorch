#pragma once

namespace knn {

namespace {
  static inline Tensor apply_loss_reduction(const Tensor& unreduced, int64_t reduction) {
    if (reduction == torch::Reduction::Mean) {
      return unreduced.mean();
    } else if (reduction == torch::Reduction::Sum) {
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
  typedef std::variant<torch::enumtype::kNone, torch::enumtype::kMean, torch::enumtype::kSum> reduction_t;
  TORCH_ARG(reduction_t, reduction) = torch::kMean;
};

struct TORCH_API BCEWithLogitsLossOptions {
  typedef std::variant<torch::enumtype::kNone, torch::enumtype::kMean, torch::enumtype::kSum> reduction_t;
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
   return torch::nn::functional::detail::binary_cross_entropy_with_logits(input, target, weight, options.reduction(), pos_weight);
  }
  BCEWithLogitsLossOptions options;
  Tensor pos_weight;
};

TORCH_MODULE(BCELoss);
TORCH_MODULE(BCEWithLogitsLoss);

} // namespace knn
