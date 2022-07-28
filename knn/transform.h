#pragma once
#include "util.h"

namespace knn {

// ------------------------------------------------------------------
// zscore - subtract mean and divide by standard deviation
// ------------------------------------------------------------------
struct TORCH_API ZscoreOptions {
 ZscoreOptions(Tensor m,Tensor d,bool b=false) : mean_(std::move(m)), stddev_(std::move(d)), inplace_(b) {}
 TORCH_ARG(Tensor, mean) = {};
 TORCH_ARG(Tensor, stddev) = {};
 TORCH_ARG(bool, inplace) = false;
};

class TORCH_API ZscoreImpl : public torch::nn::Cloneable<ZscoreImpl> {
 public:
 explicit ZscoreImpl(const ZscoreOptions& o);
 void reset() override;
 void pretty_print(std::ostream& s) const override;
 Tensor forward(Tensor t);

 ZscoreOptions options;
 Tensor mean,stddev;
};
TORCH_MODULE(Zscore);

// ---------------------------------------------------------------------------
// zscore - set/get mean,stddev & inplace flag for zscore module
//          also, routines to do the zscore given input, mean & stddev tensors
// ---------------------------------------------------------------------------
ZscoreOptions zscore(K,J,Cast);
K zscore(bool,const ZscoreOptions&);

Tensor zscore_(Tensor&,const Tensor&,const Tensor&);
Tensor zscore(const Tensor&,const Tensor&,const Tensor&);

// ------------------------------------------------------------------
// random crop - pad, then crop tensor height,width at randm offsets
// ------------------------------------------------------------------
struct TORCH_API RandomCropOptions {
 RandomCropOptions(torch::ExpandingArray<2> s,torch::ExpandingArray<4> p=0) : size_(s),pad_(p) {}

 TORCH_ARG(torch::ExpandingArray<2>, size);
 TORCH_ARG(torch::ExpandingArray<4>, pad);
 TORCH_ARG(torch::nn::functional::PadFuncOptions::mode_t, padmode) = torch::kConstant;
 TORCH_ARG(double, value) = 0;
};

class TORCH_API RandomCropImpl : public torch::nn::Cloneable<RandomCropImpl> {
 public:
 RandomCropImpl(torch::ExpandingArray<2> s,torch::ExpandingArray<4> p=0);
 explicit RandomCropImpl(const RandomCropOptions& o);
 void reset() override;
 void pretty_print(std::ostream& s) const override;
 Tensor forward(const Tensor& t);
 RandomCropOptions options;
};
TORCH_MODULE(RandomCrop);

// ------------------------------------------------------------------
// rcrop - routines to get/set options & perform the random cropping
// ------------------------------------------------------------------
RandomCropOptions rcrop(K,J,Cast);
K rcrop(bool,const RandomCropOptions&);
Tensor rcrop(const Tensor& t,const RandomCropOptions& o);

// ------------------------------------------------------------------
// random flip, horizontal or vertical depending on dimension given
// ------------------------------------------------------------------
struct TORCH_API RandomFlipOptions {
 RandomFlipOptions(double p=0.5,int64_t d=-1) : p_(p),dim_(d) {}
 TORCH_ARG(double, p);
 TORCH_ARG(int64_t, dim);
};

class TORCH_API RandomFlipImpl : public torch::nn::Cloneable<RandomFlipImpl> {
 public:
 RandomFlipImpl(double p,int64_t d);
 explicit RandomFlipImpl(const RandomFlipOptions& o);
 void reset() override;
 void pretty_print(std::ostream& s) const override;
 Tensor forward(const Tensor& t);

 RandomFlipOptions options;
};
TORCH_MODULE(RandomFlip);

// ---------------------------------------------------------------------------
// rflip - set/get probability p and dim for random horizontal/vertical flip
//       - also perform the flip given input, prob and dimension
// ---------------------------------------------------------------------------
RandomFlipOptions rflip(K,J,Cast);
K rflip(bool,const RandomFlipOptions&);
Tensor rflip(const Tensor&,const RandomFlipOptions&);

// -------------------------------------------------------------------------------
// transform: define sequential modules to transform data in training & eval mode
// -------------------------------------------------------------------------------
class TORCH_API TransformImpl : public torch::nn::Cloneable<TransformImpl> {
 public:
 TransformImpl() = default;
 void reset() override;
 void pretty_print(std::ostream& s) const override;
 void push_back(const torch::nn::Sequential& q);
 void push_back(std::string s,const torch::nn::Sequential& q);
 void push_back(const torch::nn::AnyModule& a);
 void push_back(std::string s,const torch::nn::AnyModule& a);
 Tensor forward(const Tensor& x);

 torch::nn::Sequential train = nullptr;
 torch::nn::Sequential eval  = nullptr;
};
TORCH_MODULE(Transform);

} // namespace knn
