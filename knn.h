#pragma once

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

class TORCH_API SqueezeImpl : public torch::nn::Cloneable<SqueezeImpl> {
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

class TORCH_API UnsqueezeImpl : public torch::nn::Cloneable<UnsqueezeImpl> {
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

class TORCH_API ExpandImpl : public torch::nn::Cloneable<ExpandImpl> {
 public:
 ExpandImpl(std::vector<int64_t> s) : ExpandImpl(SizeOptions(s)) {}
 explicit ExpandImpl(const SizeOptions& o) : options(o) {reset();}
 void reset() override {}
 void pretty_print(std::ostream& s) const override {s << "Expand(size=" << options.size() << ")";}
 torch::Tensor forward(const torch::Tensor& t) { return t.expand(options.size());}
 SizeOptions options;
};
TORCH_MODULE(Expand);

class TORCH_API ReshapeImpl : public torch::nn::Cloneable<ReshapeImpl> {
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

class TORCH_API CatImpl : public torch::nn::Cloneable<CatImpl> {
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

// ----------------------------------------------------------------------------------
// onehot - add convenience module for torch.nn.functional.one_hot(tensor,numclasses)
// ----------------------------------------------------------------------------------
struct TORCH_API OneHotOptions {
 OneHotOptions(int64_t n=-1) : num_classes_(n) {}
 TORCH_ARG(int64_t, num_classes);
 TORCH_ARG(c10::optional<torch::Dtype>, dtype) = c10::nullopt;
};

class TORCH_API OneHotImpl : public torch::nn::Cloneable<OneHotImpl> {
 public:
 OneHotImpl(const OneHotOptions& o) : options(o) {}
 void reset() override {}
 void pretty_print(std::ostream& s) const override {s << "OneHot(num_classes=" << options.num_classes() << ")";}
 torch::Tensor forward(const torch::Tensor& x) {
  return torch::one_hot(x,options.num_classes()).to(options.dtype() ? options.dtype().value() : torch::kFloat);
 }
 OneHotOptions options;
};
TORCH_MODULE(OneHot);

// ----------------------------------------------------------------------------------------------------
// select - add convenience module for select(dim,index)
// ----------------------------------------------------------------------------------------------------
struct TORCH_API SelectOptions {
 SelectOptions(int64_t d,int64_t i) : dim_(d),index_(i) {}
 TORCH_ARG(int64_t, dim);
 TORCH_ARG(int64_t, index);
};

class TORCH_API SelectImpl : public torch::nn::Cloneable<SelectImpl> {
 public:
 SelectImpl(const SelectOptions& o) : options(o) {}
 void reset() override {}
 void pretty_print(std::ostream& s) const override {s << "Select(dim=" << options.dim() << ",index=" << options.index() << ")";}
 torch::Tensor forward(const torch::Tensor& x) {
  return x.select(options.dim(),options.index());
 }
 SelectOptions options;
};
TORCH_MODULE(Select);

// ----------------------------------------------------------------------------------------------------
// mul - add convenience module for multiply
// ----------------------------------------------------------------------------------------------------
class TORCH_API MulImpl : public torch::nn::Cloneable<MulImpl> {
 public:
 void reset() override {}
 void pretty_print(std::ostream& s) const override {s << "Mul()";}
 torch::Tensor forward(const torch::Tensor& x,const torch::Tensor& y) {return torch::mul(x,y);}
};
TORCH_MODULE(Mul);

// ----------------------------------------------------------------------------------
// RNNOutput,LSTMOutput - convenience modules to extract 1st tensor rnn result tuples
// ----------------------------------------------------------------------------------
class TORCH_API GRUOutputImpl : public torch::nn::Cloneable<GRUOutputImpl> {
 using Tensor=torch::Tensor;
 public:
 void reset() override {}
 void pretty_print(std::ostream& s) const override {s << "GRUOutput()";}
 Tensor forward(const std::tuple<Tensor,Tensor>& x) {return std::get<0>(x);}
};
TORCH_MODULE(GRUOutput);

class TORCH_API LSTMOutputImpl : public torch::nn::Cloneable<LSTMOutputImpl> {
 using Tensor=torch::Tensor;
 public:
 void reset() override {}
 void pretty_print(std::ostream& s) const override {s << "LSTMOutput()";}
 Tensor forward(const std::tuple<Tensor,std::tuple<Tensor,Tensor>>& x) {return std::get<0>(x);}
};
TORCH_MODULE(LSTMOutput);

class TORCH_API RNNOutputImpl : public torch::nn::Cloneable<RNNOutputImpl> {
 using Tensor=torch::Tensor;
 public:
 void reset() override {}
 void pretty_print(std::ostream& s) const override {s << "RNNOutput()";}
 Tensor forward(const std::tuple<Tensor,Tensor>& x) {return std::get<0>(x);}
};
TORCH_MODULE(RNNOutput);

// ----------------------------------------------------------------------------------
// RNNFork, LSTMFork - split output & hidden state from rnn layer, apply sequential
// ----------------------------------------------------------------------------------
struct TORCH_API ForkOptions {
 ForkOptions(bool d=true) : detach_(d) {}
 TORCH_ARG(bool, detach);
};

template<typename Derived>class TORCH_API ForkBase : public torch::nn::Cloneable<Derived> {
 public:
 explicit ForkBase(const ForkOptions& o) : options(o) {seq=this->register_module("seq",seq);}
 void reset() override {}
 void push_back(std::string s,const torch::nn::AnyModule& a) {seq->push_back(s,a);}
 void push_back(const torch::nn::AnyModule& a) {push_back(c10::to_string(seq->size()),a);}
 torch::nn::Sequential seq;
 ForkOptions options;
};

class TORCH_API RNNForkImpl : public ForkBase<RNNForkImpl> {
 using Tensor=torch::Tensor;
 using Tuple=std::tuple<Tensor,Tensor>;
 public:
 explicit RNNForkImpl(const ForkOptions& o) : ForkBase<RNNForkImpl>(o) {}
 void pretty_print(std::ostream& s) const override {s << "RNNFork(detach=" << options.detach() << ")";}
 Tuple forward(const Tuple& a) {  // recieves output & hidden tensor from RNN or GRU layer
  auto x=seq->forward(std::get<0>(a));
  auto h=std::get<1>(a);
  if(options.detach())
   h.detach_();
  return std::make_tuple(x,h);
 }
};
TORCH_MODULE(RNNFork);

class TORCH_API LSTMForkImpl : public ForkBase<LSTMForkImpl> {
 using Tensor=torch::Tensor;
 using Tuple=std::tuple<Tensor,Tensor>;
 using Nested=std::tuple<Tensor,Tuple>;
 public:
 explicit LSTMForkImpl(const ForkOptions& o) : ForkBase<LSTMForkImpl>(o) {}
 void pretty_print(std::ostream& s) const override {s << "LSTMFork(detach=" << options.detach() << ")";}
 Nested forward(const Nested& a) {
  auto x=seq->forward(std::get<0>(a));
  auto h=std::get<1>(a);
  if(options.detach())
   std::get<0>(h).detach_(),std::get<1>(h).detach_();
  return std::make_tuple(x,h);
 }
};
TORCH_MODULE(LSTMFork);

// ----------------------------------------------------------------------------------
// SeqNest - derived from Sequential to allow nested sequentials 
//         - no templatized forward result means can be stored as an AnyModule
//         - forward method accepts up to three tensors x,y,z w'y & z optional
//           forward result is tensor only
// ---------------------------------------------------------------------------------
class TORCH_API SeqNestImpl : public torch::nn::SequentialImpl {
  public:
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

// --------------------------------------------------------------------------------------------------
// SeqJoin - define sequential modules for inputs x & y w'layer for joining the output of each module
// --------------------------------------------------------------------------------------------------
class TORCH_API SeqJoinImpl : public torch::nn::Cloneable<SeqJoinImpl> {
 public:
 SeqJoinImpl() = default;
 void reset() override {}
 void pretty_print(std::ostream& s) const override {s << "SeqJoin";}

 void push_back(const torch::nn::Sequential& q) {
  push_back(children().size()==0 ? "qx" : "qy", q);
 }

 void push_back(std::string s,const torch::nn::Sequential& q) {
  if(children().size()==0)
   qx=register_module(s,std::move(q));
  else if(children().size()==1)
   qy=register_module(s,std::move(q));
  else
   AT_ERROR("seqjoin: both sequential layers already defined");
 }

 void push_back(const torch::nn::AnyModule& a) {
  push_back("join",a);
 }

 void push_back(std::string s,const torch::nn::AnyModule& a) {
  TORCH_CHECK(children().size()==2, "seqjoin: both sequential layers must be defined first");
  TORCH_CHECK(join.is_empty(), "seqjoin: join layer already defined");
  join=std::move(a);
  register_module(s,join.ptr());
 }

 torch::Tensor forward(const torch::Tensor& x,const torch::Tensor& y) {
  TORCH_CHECK(!join.is_empty(), "seqjoin: join layer not defined");
  return join.forward(qx.is_empty() || !qx->children().size() ? x : qx->forward(x),
                      qy.is_empty() || !qy->children().size() ? y : qy->forward(y));
 }
 torch::nn::Sequential qx = nullptr;
 torch::nn::Sequential qy = nullptr;
 torch::nn::AnyModule  join;
};
TORCH_MODULE(SeqJoin);

// ----------------------------------------------------------------------
// SeqRNN
// ----------------------------------------------------------------------
/*
  if(LSTM) {
   auto a=rnn->forward(x,h);
   x=std::get<0>(a);
   y=std::get<0>(std::get<1>(a));
   z=std::get<1>(std::get<1>(a));
  } else {
   auto a=rnn->forward(x,h);
   x=std::get<0>(a);
   y=std::get<1>(a);
  }
  x=seq(x);
  if(hidden) {
   if(detach) {y.detach_(); if(z.defined()) z.detach_();}
   v.push_back(x);
   v.push_back(y);
   v.push_back(z);
   return v
  } else {
   return {x};
  }
*/

// ----------------------------------------------------------------------
// generic module to accept tensor parameters, buffers and child modules
// ----------------------------------------------------------------------
class TORCH_API BaseModuleImpl : public torch::nn::Cloneable<BaseModuleImpl> {
 public:
 BaseModuleImpl() = default;
 void reset() override {}
 void pretty_print(std::ostream& s) const override {s << "BaseModule";}
 torch::Tensor forward(const torch::Tensor& x) {AT_ERROR("nyi");}
};
TORCH_MODULE(BaseModule);

// ----------------------------------------------------------------------
// ModuleDict - defined here in antcipation of release in 1.7.1 or 1.8
// ----------------------------------------------------------------------
namespace torch {
namespace nn {

class TORCH_API ModuleDictImpl : public Cloneable<ModuleDictImpl> {
 public:
  using Iterator = torch::OrderedDict<std::string, std::shared_ptr<Module>>::Iterator;
  using ConstIterator = torch::OrderedDict<std::string, std::shared_ptr<Module>>::ConstIterator;

  ModuleDictImpl() = default;
  explicit ModuleDictImpl(const std::vector<std::pair<std::string, std::shared_ptr<Module>>>& modules) {update(modules);}
  explicit ModuleDictImpl(const torch::OrderedDict<std::string, std::shared_ptr<Module>>& modules) {update(modules);}

  std::vector<std::pair<std::string, std::shared_ptr<Module>>> items() const {return modules_.pairs();}
  std::vector<std::string> keys() const {return modules_.keys();}
  std::vector<std::shared_ptr<Module>> values() const {return modules_.values();}
  Iterator begin() {return modules_.begin();}
  ConstIterator begin() const {return modules_.begin();}
  Iterator end() {return modules_.end();}
  ConstIterator end() const {return modules_.end();}
  size_t size() const noexcept {return modules_.size();}
  bool empty() const noexcept {return modules_.is_empty();}
  bool contains(const std::string& key) const noexcept {return modules_.contains(key);}
  void clear() {modules_.clear();}

  std::shared_ptr<Module> clone(
    const optional<Device>& device = nullopt) const override {
    auto clone = std::make_shared<ModuleDictImpl>();
    for (const auto& module : modules_) {
      clone->insert(module.key(), module.value()->clone(device));
    }
    return clone;
  }

  void reset() override {}
  void pretty_print(std::ostream& stream) const override {stream << "torch::nn::ModuleDict";}

  std::shared_ptr<Module> operator[](const std::string& key) const {return modules_[key];}

  template <typename T>
  T& at(const std::string& key) {
    static_assert(
        torch::detail::is_module<T>::value,
        "Can only call ModuleDict::at with an nn::Module type");
    return *modules_[key]->as<T>();
  }

  template <typename T>
  const T& at(const std::string& key) const {
    static_assert(
        torch::detail::is_module<T>::value,
        "Can only call ModuleDict::at with an nn::Module type");
    return *modules_[key]->as<T>();
  }

  std::shared_ptr<Module> pop(const std::string& key) {
    auto module = modules_[key];
    modules_.erase(key);
    // Not remove the registration of the module to make it consistent with python version.
    return module;
  }

  /// Updated the `ModuleDict` with a vector of key-module pairs.
  void update(
      const std::vector<std::pair<std::string, std::shared_ptr<Module>>>& modules) {
    for (auto& item : modules) {
      insert(item.first, item.second);
    }
  }

  /// Updated the `ModuleDict` with key-value pairs from `OrderedDict` or `ModuleDict`.
  template <typename Container>
  void update(const Container& container) {
    for (auto& item : container) {
      insert(item.key(), item.value());
    }
  }

private:
  torch::OrderedDict<std::string, std::shared_ptr<Module>> modules_;

  void insert(const std::string& key, std::shared_ptr<Module> module) {
    if (contains(key)) {
      modules_[key] = std::move(module);
      replace_module(key, modules_[key]);
    }
    else {
      modules_.insert(key, std::move(module));
      register_module(key, modules_.back().value());
    }
  }

};

TORCH_MODULE(ModuleDict);

} // namespace nn
} // namespace torch
