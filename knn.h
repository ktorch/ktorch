#pragma once

std::string mlabel(const std::shared_ptr<torch::nn::Module>&);
torch::Tensor zscore_(torch::Tensor&,const torch::Tensor&,const torch::Tensor& d);
torch::Tensor zscore(const torch::Tensor&,const torch::Tensor&,const torch::Tensor&);
torch::Tensor randomcrop(const torch::Tensor&,int64_t,int64_t,const torch::Tensor&);

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
 CatImpl(const CatOptions& o) : options(o) {reset();}
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
 OneHotImpl(const OneHotOptions& o) : options(o) {reset();}
 void reset() override {}
 void pretty_print(std::ostream& s) const override {s << "OneHot(num_classes=" << options.num_classes() << ")";}
 torch::Tensor forward(const torch::Tensor& x) {
  return torch::one_hot(x,options.num_classes()).to(options.dtype() ? options.dtype().value() : torch::kFloat);
 }
 OneHotOptions options;
};
TORCH_MODULE(OneHot);

// ------------------------------------------------------------------------------------------
// index - add convenience module for tensor.select(dim,ind) or index_select(tensor,dim,ind)
// ------------------------------------------------------------------------------------------
struct TORCH_API IndexOptions {
 IndexOptions(int64_t d,int64_t i) : dim_(d),ind_(torch::full({},torch::Scalar(i))) {}
 IndexOptions(int64_t d,torch::Tensor i) : dim_(d),ind_(i) {}
 TORCH_ARG(int64_t, dim);
 TORCH_ARG(torch::Tensor, ind);
};

class TORCH_API IndexImpl : public torch::nn::Cloneable<IndexImpl> {
 public:
 IndexImpl(const IndexOptions& o) : options(o) {reset();}
 void reset() override {
  TORCH_CHECK(options.ind().dtype() == torch::kLong, "select: long(s) expected for indices, ",options.ind().dtype(),"(s) supplied");
  TORCH_CHECK(options.ind().dim()<2, "select: single index or list expected, ",options.ind().dim(),"-d tensor supplied");
  ind=register_buffer("ind", options.ind());
 }
 void pretty_print(std::ostream& s) const override {s << "Index(dim=" << options.dim() << ",ind=" << options.ind() << ")";}
 torch::Tensor forward(const torch::Tensor& x) {
  if(ind.dim())
   return torch::index_select(x,options.dim(),ind);
  else
   return x.select(options.dim(),ind.item().toLong());
 }
 IndexOptions options;
 torch::Tensor ind;
};
TORCH_MODULE(Index);

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

// ---------------------------------------------------------------------------------------
// fork - create two branches with AnyModule/Sequential to separately process input tensor
// ---------------------------------------------------------------------------------------
class TORCH_API ForkImpl : public torch::nn::Cloneable<ForkImpl> {
 using Tensor=torch::Tensor;
 using Tuple=std::tuple<Tensor,Tensor>;
 public:
 void reset() override {}
 void pretty_print(std::ostream& s) const override {s << "Fork";}

 void push_back(std::string s,const torch::nn::AnyModule& m) {
  if(a.is_empty() && qa.is_empty()) 
   a=std::move(m), register_module(s.size() ? s : "a", m.ptr());
  else if(b.is_empty() && qb.is_empty())
   b=std::move(m), register_module(s.size() ? s : "b", m.ptr());
  else
   TORCH_CHECK(false, "fork: cannot add ",mlabel(m.ptr())," module, both left & right forks already defined");
 }
 
 void push_back(const torch::nn::AnyModule& m) {push_back(std::string(),m);}

 void push_back(std::string s,const torch::nn::Sequential& q) {
  if(a.is_empty() && qa.is_empty()) 
   qa=register_module(s.size() ? s : "qa", q);
  else if(b.is_empty() && qb.is_empty())
   qb=register_module(s.size() ? s : "qb", q);
  else
   TORCH_CHECK(false, "fork: cannot add ",mlabel(q.ptr())," module, both left & right forks already defined");
 }

 void push_back(const torch::nn::Sequential& q) {push_back(std::string(),q);}

 Tuple forward(const Tensor& x) {
  Tensor y=a.is_empty() ? (qa.is_empty() ? x : qa->forward(x)) : a.forward(x);
  Tensor z=b.is_empty() ? (qb.is_empty() ? x : qb->forward(x)) : b.forward(x);
  return std::make_tuple(y,z);
 }

 torch::nn::AnyModule a,b;
 torch::nn::Sequential qa=nullptr,qb=nullptr;
};
TORCH_MODULE(Fork);

// --------------------------------------------------------------------------------
// Recur - receive input & hidden state for rnn layer
//         sequential modules in/out apply transformations to input & rnn output
// --------------------------------------------------------------------------------
struct TORCH_API RecurOptions {
 RecurOptions(bool d=true) : detach_(d) {}
 TORCH_ARG(bool, detach);
};

class TORCH_API RecurImpl : public torch::nn::Cloneable<RecurImpl> {
 using Tensor=torch::Tensor;
 using Tuple=std::tuple<Tensor,Tensor>;
 using OptTuple=c10::optional<Tuple>;
 using Nested=std::tuple<Tensor,Tuple>;
 public:
 explicit RecurImpl(const RecurOptions& o) : options(o) {reset();}

 void reset() override {}
 void pretty_print(std::ostream& s) const override {s << "Recur";}

 void push_back(const std::shared_ptr<Module>& m) {
  bool e=lstm.is_empty() && gru.is_empty() && rnn.is_empty();
  if(const auto& a=std::dynamic_pointer_cast<torch::nn::SequentialImpl>(m)) {
    if(e) {
     TORCH_CHECK(in.is_empty(), "recur: cannot add a sequential module for output processing until a recurrent module(lstm,gru,rnn) defined");
     in=register_module("in", torch::nn::Sequential(a));
    } else {
     TORCH_CHECK(out.is_empty(), "recur: sequential module for output processing already added, cannot add another sequential module");
     out=register_module("out", torch::nn::Sequential(a));
    }
  } else if(const auto& a=std::dynamic_pointer_cast<torch::nn::LSTMImpl>(m)) {
   TORCH_CHECK(e, "recur: cannot add lstm module, ",(gru.is_empty() ? "rnn" : "gru")," module already defined");
   lstm=register_module("lstm",torch::nn::LSTM(a));
  } else if(const auto& a=std::dynamic_pointer_cast<torch::nn::GRUImpl>(m)) {
   TORCH_CHECK(e, "recur: cannot add gru module, ",(rnn.is_empty() ? "lstm" : "rnn")," module already defined");
   gru=register_module("gru",torch::nn::GRU(a));
  } else if(const auto& a=std::dynamic_pointer_cast<torch::nn::RNNImpl>(m)) {
   TORCH_CHECK(e, "recur: cannot add rnn module, ",(gru.is_empty() ? "lstm" : "gru")," module already defined");
   rnn=register_module("rnn",torch::nn::RNN(a));
  } else {
   TORCH_CHECK(false, 
               "recur: unable to add ",mlabel(m),
               ", expecting sequential modules for input/output processing or recurrent module(lstm,gru,rnn)");
  }
 }

 std::vector<Tensor> forward(const Tensor& x,const Tensor& y,const Tensor& z) {
  std::vector<Tensor> v;
  if(!lstm.is_empty()) {
    Nested r=z.defined() ? lstm->forward(in->is_empty() ? x : in->forward(x), OptTuple(std::make_tuple(y,z)))
                         : lstm->forward(in->is_empty() ? x : in->forward(x));
    Tuple& h=std::get<1>(r);
    if(options.detach()) std::get<0>(h).detach_(), std::get<1>(h).detach_();
    v.push_back(out->is_empty() ? std::get<0>(r) : out->forward(std::get<0>(r)));
    v.push_back(std::get<0>(h));
    v.push_back(std::get<1>(h));
  } else if(!gru.is_empty() || !rnn.is_empty()) {
    Tuple r=rnn.is_empty() ? gru->forward(in->is_empty() ? x : in->forward(x), y)
                           : rnn->forward(in->is_empty() ? x : in->forward(x), y);
    if(options.detach()) std::get<1>(r).detach_();
    v.push_back(out->is_empty() ? std::get<0>(r) : out->forward(std::get<0>(r)));
    v.push_back(std::get<1>(r));
  } else {
   TORCH_CHECK(false, "recur: no recurrent lstm, gru or rnn module defined");
  }
  return v;
 }

 RecurOptions options;
 torch::nn::Sequential in=nullptr;
 torch::nn::LSTM lstm=nullptr;
 torch::nn::GRU   gru=nullptr;
 torch::nn::RNN   rnn=nullptr;
 torch::nn::Sequential out=nullptr;
};
TORCH_MODULE(Recur);

// ---------------------------------------------------------------------------------------
// residual - create up to two Sequentials and an optional activation function
// ---------------------------------------------------------------------------------------
class TORCH_API ResidualImpl : public torch::nn::Cloneable<ResidualImpl> {
 public:
 void reset() override {}
 void pretty_print(std::ostream& s) const override {s << "Residual";}

 torch::Tensor forward(const torch::Tensor& x) {
  TORCH_CHECK(!q1.is_empty(), "residual: no modules defined for forward calculation");
  if(q2.is_empty())
   return fn.is_empty() ? q1->forward(x) + x : fn.forward(q1->forward(x) + x);
  else
   return fn.is_empty() ? q1->forward(x) + q2->forward(x) : fn.forward(q1->forward(x) + q2->forward(x));
 }

 void push_back(std::string s,const torch::nn::Sequential& q) {
 if(q1.is_empty())
  q1=register_module(s.size() ? s : "q1", q);
 else if(q2.is_empty() && fn.is_empty())
  q2=register_module(s.size() ? s : "q2", q);
 else
  TORCH_CHECK(false, "residual: ",
             (q2.is_empty() ? "activation function already defined, cannot add a 2nd sequential module" 
                            : "both sequential modules already defined, cannot add another sequential module"));
 }

 void push_back(const torch::nn::Sequential& q) {push_back(std::string(),q);}

 void push_back(std::string s,const torch::nn::AnyModule& m) {
  TORCH_CHECK(!q1.is_empty(), "residual: cannot add ", mlabel(m.ptr()), " module until sequential module(s) defined");
  TORCH_CHECK( fn.is_empty(), "residual: cannot add ", mlabel(m.ptr()), " module, activation function already defined");
  fn=std::move(m), register_module(s.size() ? s : "fn", m.ptr());
 }

 void push_back(const torch::nn::AnyModule& m) {push_back(std::string(),m);}

 torch::nn::Sequential q1=nullptr,q2=nullptr;
 torch::nn::AnyModule fn;
};
TORCH_MODULE(Residual);

// ----------------------------------------------------------------------------------
//  nbeats - container for N-BEATS model for forecasting
//         = a ModuleList for processing blocks (generic/seasonal/trend)
// ----------------------------------------------------------------------------------
class TORCH_API NBeatsImpl : public torch::nn::ModuleListImpl {
 using ModuleListImpl::ModuleListImpl;
 using Tensor=torch::Tensor;
 using Tuple=std::tuple<Tensor,Tensor>;
 public:
 void pretty_print(std::ostream& stream) const override {stream << "NBeats";}

 Tensor forward1(Tensor x) {
  Tensor y; size_t i=0;
  for(auto& a:children()) {
   auto *q=a->as<torch::nn::Sequential>();
   TORCH_CHECK(q, "nbeats: unexpected ",mlabel(a)," module in block[",i,"], expecting sequential block");
   Tensor b,f;
   std::tie(b,f)=q->forward<Tuple>(x);
   x=x-b; y=y.defined() ? y+f : f; i++;
  }
  return y;
 }

 void blockforward(std::shared_ptr<torch::nn::Module>& m,Tensor& x,Tensor& y) {
  if(auto *a=m->as<torch::nn::Sequential>()) {
   Tensor b,f; std::tie(b,f)=a->forward<Tuple>(x);
   x=x-b; y=y.defined() ? y+f : f;
  } else if(auto *a=m->as<torch::nn::ModuleList>()) {
   for(auto& q:a->children())
    blockforward(q,x,y);
  } else {
   TORCH_CHECK(false,"nbeats: not a sequential or list module");
  }
 }
  
 Tensor forward(Tensor x) {
  Tensor y; size_t i=0;
  for(auto& m:children())
   blockforward(m,x,y), i++;
  return y;
 }
};
TORCH_MODULE(NBeats);

// ----------------------------------------------------------------------------------
//  Eval - define AnyModule/Sequential for additional evaluation calcs on output
// ----------------------------------------------------------------------------------
class TORCH_API EvalImpl : public torch::nn::Cloneable<EvalImpl> {
 public:
 //explicit EvalImpl(const EvalOptions& o) : options(o) {}

 void reset() override {}
 void pretty_print(std::ostream& s) const override {s << "Eval";}

 torch::Tensor forward(const torch::Tensor& x) {
  if(is_training() || (any.is_empty() && seq.is_empty()))
   return x;
  else if(!any.is_empty())
   return any.forward(x);
  else
   return seq->forward(x);
 }

 torch::nn::AnyModule  any;
 torch::nn::Sequential seq=nullptr;
};
TORCH_MODULE(Eval);


// ----------------------------------------------------------------------------------
// SeqNest - derived from Sequential to allow nested sequentials 
//         - no templatized forward result means can be stored as an AnyModule
//         - forward method accepts up to three tensors x,y,z w'y & z optional
//           forward result is tensor only
// ---------------------------------------------------------------------------------
class TORCH_API SeqNestImpl : public torch::nn::SequentialImpl {
  public:
  using SequentialImpl::SequentialImpl;

  void pretty_print(std::ostream& stream) const override {
    stream << "SeqNest";
  }

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
  TORCH_CHECK(children().size()<2, "seqjoin: both sequential layers already defined");
  if(children().size()==0)
   qx=register_module(s,std::move(q));
  else
   qy=register_module(s,std::move(q));
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
// generic module to accept tensor parameters, buffers and child modules
// ----------------------------------------------------------------------
class TORCH_API BaseModuleImpl : public torch::nn::Cloneable<BaseModuleImpl> {
 public:
 BaseModuleImpl() = default;
 void reset() override {}
 void pretty_print(std::ostream& s) const override {s << "BaseModule";}
 torch::Tensor forward(const torch::Tensor& x) {TORCH_CHECK(false,"nyi");}
};
TORCH_MODULE(BaseModule);

// ------------------------------------------------------------------
// zscore - subtract mean and divide by standard deviation
// ------------------------------------------------------------------
struct TORCH_API ZscoreOptions {
 ZscoreOptions(torch::Tensor m,torch::Tensor d,bool b=false) : mean_(std::move(m)), stddev_(std::move(d)), inplace_(b) {}
 TORCH_ARG(torch::Tensor, mean) = {};
 TORCH_ARG(torch::Tensor, stddev) = {};
 TORCH_ARG(bool, inplace) = false;
};

class TORCH_API ZscoreImpl : public torch::nn::Cloneable<ZscoreImpl> {
 public:
 explicit ZscoreImpl(const ZscoreOptions& o) : options(o) {reset();}
 void reset() override {
  TORCH_CHECK(options.mean().defined()   && options.mean().is_floating_point(),   "zscore: mean(s) not defined as float/double");
  TORCH_CHECK(options.stddev().defined() && options.stddev().is_floating_point(), "zscore: std deviation(s) not defined as float/double");
 }

 void pretty_print(std::ostream& s) const override {
  auto f=[](auto& x) {return torch::ArrayRef<double>(x.to(torch::kDouble).template data_ptr<double>(),x.numel()>3 ? 3 : x.numel()).vec();};
  s << std::boolalpha
    << "Zscore(mean=" << f(options.mean()) << ", stddev=" << f(options.stddev()) << ", inplace=" << options.inplace() << ")";
 }

 torch::Tensor forward(torch::Tensor t) {
  return options.inplace() ? zscore_(t,options.mean(),options.stddev()) : zscore(t,options.mean(),options.stddev());
 }
  
 ZscoreOptions options;
};
TORCH_MODULE(Zscore);

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
 RandomCropImpl(torch::ExpandingArray<2> s,torch::ExpandingArray<4> p=0) : RandomCropImpl(RandomCropOptions(s,p)) {}
 explicit RandomCropImpl(const RandomCropOptions& o) : options(o) {reset();}

 void reset() override {
  p=this->register_buffer("p", torch::empty(1, torch::kLong));
 }

 void pretty_print(std::ostream& s) const override {
  s << "RandomCrop(size=" << options.size();
  if(*options.pad() != *torch::ExpandingArray<4>(0)) s << ", pad=" << options.pad();
  s << ")";
 }

 torch::Tensor forward(const torch::Tensor& t) {
  return(*options.pad() == *torch::ExpandingArray<4>(0) ? t
         : torch::nn::functional::detail::pad(t,options.pad(),options.padmode(),options.value()),
         (*options.size())[0],
         (*options.size())[1],
          p);
 }

 RandomCropOptions options;
 torch::Tensor p;  //tensor used for random top,left corner
};
TORCH_MODULE(RandomCrop);

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
 RandomFlipImpl(double p,int64_t d) : RandomFlipImpl(RandomFlipOptions(p,d)) {}
 explicit RandomFlipImpl(const RandomFlipOptions& o) : options(o) {reset();}

 void reset() override {
  p=this->register_buffer("p", torch::empty(1, torch::kDouble));
 }

 void pretty_print(std::ostream& s) const override {
  s << "RandomFlip(p=" << options.p() << ", dim=" << options.dim() << ")";
 }

 torch::Tensor forward(const torch::Tensor& t) {
  return p.uniform_(0,1).item().toDouble()<options.p() ? t.flip(options.dim()) : t;
 }

 RandomFlipOptions options;
 torch::Tensor p;
};
TORCH_MODULE(RandomFlip);
