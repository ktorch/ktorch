#pragma once

#ifdef __clang__
# pragma clang diagnostic push
# pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"    // FORWARD_HAS_DEFAULT_ARGS
#endif

// --------------------------------------------------------------------------
// general pad: create module to match functional call with size, mode, value
// --------------------------------------------------------------------------
class TORCH_API PadImpl : public torch::nn::Cloneable<PadImpl> {
 public:
  PadImpl(std::vector<int64_t> p) : PadImpl(torch::nn::functional::PadFuncOptions(p)) {}
  explicit PadImpl(const torch::nn::functional::PadFuncOptions& o) : options(o) {reset();}
  void reset() override {}
  torch::Tensor forward(const torch::Tensor& input) {
   return torch::nn::functional::pad(input,options);
  }
  torch::nn::functional::PadFuncOptions options;
};

TORCH_MODULE(Pad);

// -------------------------------------------------------------
//  squeeze - remove dimension(s) from tensor
//  unsqueeze - add dimension to tensor
// -------------------------------------------------------------
struct TORCH_API SqueezeOptions {
 SqueezeOptions(int64_t d,bool b=false) : dim_(d),inplace_(b) {}
 SqueezeOptions() {}
 TORCH_ARG(c10::optional<int64_t>, dim) = c10::nullopt;
 TORCH_ARG(bool, inplace) = false;
};

class SqueezeImpl : public torch::nn::Cloneable<SqueezeImpl> {
 public:
  SqueezeImpl(int64_t d,bool b=false) : SqueezeImpl(SqueezeOptions(d,b)) {}
  SqueezeImpl() : SqueezeImpl(SqueezeOptions()) {}
  explicit SqueezeImpl(const SqueezeOptions& o) : options(o) {reset();}

  void reset() override {}

  void pretty_print(std::ostream& s) const override {
   s << "Squeeze(dim="; options.dim() ? s << options.dim().value() : s << "None";
   s << ", inplace=" << options.inplace() <<")";
  }

  torch::Tensor forward(const torch::Tensor& t) {
   if(options.dim().has_value()) {
    if(options.inplace())
     return t.squeeze_(options.dim().value());
    else
     return t.squeeze(options.dim().value());
   } else {
    if(options.inplace())
     return t.squeeze_();
    else
     return t.squeeze();
   }
  };
  SqueezeOptions options;
};
TORCH_MODULE(Squeeze);

class UnsqueezeImpl : public torch::nn::Cloneable<UnsqueezeImpl> {
 public:
  UnsqueezeImpl(int64_t d,bool b=false) : UnsqueezeImpl(SqueezeOptions(d,b)) {}
  explicit UnsqueezeImpl(const SqueezeOptions& o) : options(o) {reset();}

  void reset() override {TORCH_CHECK(options.dim().has_value(),"unsqueeze: no dimension given");}

  void pretty_print(std::ostream& s) const override {
   s << "Unsqueeze(dim="; options.dim() ? s << options.dim().value() : s << "None";
   s << ", inplace=" << options.inplace() <<")";
  }

  torch::Tensor forward(const torch::Tensor& t) {
   if(options.inplace())
    return t.unsqueeze_(options.dim().value());
   else
    return t.unsqueeze(options.dim().value());
  };
  SqueezeOptions options;
};
TORCH_MODULE(Unsqueeze);

// -------------------------------------------------------------
// expand & reshape - modules with size options
// -------------------------------------------------------------
struct TORCH_API SizeOptions {
 SizeOptions(std::vector<int64_t> s) : size_(std::move(s)) {}
 TORCH_ARG(std::vector<int64_t>, size);
};

class ExpandImpl : public torch::nn::Cloneable<ExpandImpl> {
 public:
 ExpandImpl(std::vector<int64_t> s) : ExpandImpl(SizeOptions(s)) {}
 explicit ExpandImpl(const SizeOptions& o) : options(o) {reset();}
 void reset() override {}
 void pretty_print(std::ostream& s) const override {s << "Expand(size=" << options.size() << ")";}
 torch::Tensor forward(const torch::Tensor& t) { return t.expand(options.size());}
 SizeOptions options;
};
TORCH_MODULE(Expand);

class ReshapeImpl : public torch::nn::Cloneable<ReshapeImpl> {
 public:
 ReshapeImpl(std::vector<int64_t> s) : ReshapeImpl(SizeOptions(s)) {}
 explicit ReshapeImpl(const SizeOptions& o) : options(o) {reset();}
 void reset() override {}
 void pretty_print(std::ostream& s) const override {s << "Reshape(size=" << options.size() << ")";}
 torch::Tensor forward(const torch::Tensor& t) { return t.reshape(options.size());}
 SizeOptions options;
};
TORCH_MODULE(Reshape);

// ----------------------------------------------------------------------------------------------------
// cat - add convenience module for cat(tensors,dim)
// ----------------------------------------------------------------------------------------------------
struct TORCH_API CatOptions {
 CatOptions(int64_t d=0) : dim_(d) {}
 TORCH_ARG(int64_t, dim);
};

class CatImpl : public torch::nn::Cloneable<CatImpl> {
 public:
 CatImpl(const CatOptions& o) : options(o) {}
 void reset() override {}
 void pretty_print(std::ostream& s) const override {s << "Cat(dim=" << options.dim() << ")";}
 torch::Tensor forward(const torch::Tensor& x,const torch::Tensor& y) {
  return torch::cat({x,y},options.dim());
 }
 CatOptions options;
};
TORCH_MODULE(Cat);


// --------------------------------------------------------------------------------------------------
// SeqJoin - define sequential modules for inputs x & y w'layer for joining the output of each module
// --------------------------------------------------------------------------------------------------
class TORCH_API SeqJoinImpl : public torch::nn::Cloneable<SeqJoinImpl> {
 public:
 SeqJoinImpl() = default;
 void reset() override {}
 void pretty_print(std::ostream& s) const override {s << "SeqJoin";}

 void push_back(torch::nn::Sequential q) {
  push_back(children().size()==0 ? "qx" : "qy", q);
 }

 void push_back(std::string s,torch::nn::Sequential q) {
  if(children().size()==0)
   qx=register_module(s,std::move(q));
  else if(children().size()==1)
   qy=register_module(s,std::move(q));
  else
   AT_ERROR("both sequential layers already defined");
 }

 void push_back(torch::nn::AnyModule a) {
  push_back("join",a);
 }

 void push_back(std::string s,torch::nn::AnyModule a) {
  TORCH_CHECK(children().size()==2, "Both sequential layers must be defined first");
  TORCH_CHECK(join.is_empty(), "join layer already defined");
  join=std::move(a);
  register_module(s,join.ptr());
 }

 Tensor forward(const Tensor& x,const Tensor& y) {
  TORCH_CHECK(!join.is_empty(), "join layer not defined");
  return join.forward(qx.is_empty() || !qx->children().size() ? x : qx->forward(x),
                      qy.is_empty() || !qy->children().size() ? y : qy->forward(y));
 }
 Sequential qx = nullptr;
 Sequential qy = nullptr;
 AnyModule  join;
};
TORCH_MODULE(SeqJoin);

// ----------------------------------------------------------------------------------
// SeqNest - derived from Sequential to allow nested sequentials 
//         - no templatized forward result means can be stored as an AnyModule
//         - forward method accepts up to three tensors x,y,z w'y & z optional
// ---------------------------------------------------------------------------------
struct TORCH_API SeqNestImpl : public torch::nn::SequentialImpl {
  using SequentialImpl::SequentialImpl;

  torch::Tensor forward(const torch::Tensor& x,const torch::Tensor& y={},const torch::Tensor& z={}) {
   if(y.defined())
    return z.defined() ? SequentialImpl::forward(x,y,z) : SequentialImpl::forward(x,y);
   else
    return SequentialImpl::forward(x);
  }
 protected:
  FORWARD_HAS_DEFAULT_ARGS({1, torch::nn::AnyValue(torch::Tensor())}, {2, torch::nn::AnyValue(torch::Tensor())})
};
TORCH_MODULE(SeqNest);

// ---------------------------------------------------------------------------
// Layer - variant to hold different layer types, containers & generic modules
// ---------------------------------------------------------------------------
using Layer=c10::variant<Sequential,SeqNest,SeqJoin,AnyModule,NamedAnyModule>;
enum class Layers {sequential,seqnest,seqjoin,any,anyname};

#ifdef __clang__
# pragma clang diagnostic pop
#endif
