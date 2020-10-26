#include "ktorch.h"

KAPI f(K x) {
 KTRY
  BaseModule m;
  m->register_module("linear",torch::nn::Linear(1,2));
  m->register_parameter("tensor",torch::randn(10));
  auto a=AnyModule(m);
  std::cerr << *m << "\n";
  //return kdict(m->named_parameters());
  return mget(true,true,*m);
 KCATCH("xmodule")
}

KAPI g(K x) {
 K r=ktn(KJ,6);
 torch::optim::Adagrad g(std::vector<torch::optim::OptimizerParamGroup>{}); kJ(r)[0]=g.param_groups().size();
 torch::optim::Adam    a(std::vector<torch::optim::OptimizerParamGroup>{}); kJ(r)[1]=a.param_groups().size();
 torch::optim::AdamW   w(std::vector<torch::optim::OptimizerParamGroup>{}); kJ(r)[2]=w.param_groups().size();
 torch::optim::LBFGS   l(std::vector<Tensor>{}); l.param_groups().clear();  kJ(r)[3]=l.param_groups().size();
 torch::optim::RMSprop m(std::vector<torch::optim::OptimizerParamGroup>{}); kJ(r)[4]=m.param_groups().size();
 torch::optim::SGD     s(std::vector<torch::optim::OptimizerParamGroup>{},torch::optim::SGDOptions(.01)); kJ(r)[5]=s.param_groups().size();
 //return r;
 return optstate(true,true,Cast::adam,a);
}

/*
KAPI cb(K x,K y) {
 KTRY
  auto *l=xloss(x);
  TORCH_CHECK(l, "need loss");
  TORCH_CHECK(y->t==-KS, "need symbol for callback");
  l->cb=y->s;
  return(K)0;
 KCATCH("callback");
}

KAPI fw(K x) {
 KTRY
  auto *l=xloss(x);
  TORCH_CHECK(l, "need loss");
  TORCH_CHECK(l->cb.size(), "no callback");
  return k(0,(S)l->cb.c_str(),r1(x),0);
 KCATCH("callback");
}
*/

KAPI parms(K x) {
 KTRY
  auto *g=xtag(x);
  TORCH_CHECK(g, "not implemented for ",kname(x));
  switch(g->a) {
   case Class::module: 
   case Class::model:
    return kdict(mref(g).named_parameters());
   default:
     AT_ERROR("nyi");
  }
 KCATCH("parms/buffers");
}

KAPI optdefaults(K x) {
using Adagrad        = torch::optim::Adagrad;
using AdagradOptions = torch::optim::AdagradOptions;
 auto o=AdagradOptions(1.0);
 std::vector<torch::Tensor> v;
 for (size_t i = 0; i < 3; i++)
  v.push_back(torch::randn(10));
 auto a = Adagrad(v);
 a.param_groups()[0].set_options(std::make_unique<AdagradOptions>(o));
 a.defaults()=o;
 // test for defaults() method with non-const reference
 auto& d=static_cast<AdagradOptions&>(a.defaults());
 std::cerr << "defaults match specified options: " << (d == o) << "\n";
 std::cerr << "defaults match default options: " << (d == AdagradOptions()) << "\n";
 return (K)0;
}

/*
bool msimple(K x) {
 if(!x->t && x->n>1 && kK(x)[0]->t == -KS)
  return container(msym(kK(x)[0]->s)) ? xnone(x,x->n-1) : true;
 else
  return false;
}
*/
KAPI mrefs(K x) {
 torch::nn::Tanh f;
 std::cerr << "f ref count: " << f.ptr().use_count() << "\n";
 {
  //torch::nn::AnyModule a(f);
  //std::cerr << "f ref count: " << f.ptr().use_count() << "\n";
  torch::nn::Sequential q((AnyModule(f)));
  std::cerr << "f ref count: " << f.ptr().use_count() << "\n";
 }
 std::cerr << "f ref count: " << f.ptr().use_count() << "\n";
 return (K)0;
}

static bool parmname(const Tensor& t,const Module& m,S s) {
  S nm=mname(m);
  for(auto& p:m.named_parameters()) {
   if(t.is_same(p.value())) {
    std::cerr << "optimizer parameter: " << (nm ? nm : "") << "." << p.key() << "\n";
    return true;
   }
  }
 return false;
}


static void optgroup1(const Optimizer& o,const Module &m) {
S s=nullptr;
 J i=0;
 std::cerr << "state size: " << o.state().size() << "\n";
 std::cerr << "groups: " << o.param_groups().size() << "\n";
 for(auto& g:o.param_groups()) {
  std::cerr << "group[" << i << "] count: " << g.params().size() << "\n";
  std::cerr << "has options: " << g.has_options() << "\n";
  for(auto& p:g.params()) {
   if(!parmname(p,m,s))
    std::cerr << "failed to find parameter\n";
   }
  }
}

KAPI optgroup(K x) {
 KTRY
  TORCH_CHECK(!x->t && x->n==2,"expecting 2-element general list");
  Kmodule *q=xmodule(x,0);
  Kopt *o=xoptim(x,1);
  TORCH_CHECK(q && o, "expecting module & optimizer");
  auto& m=mref(q);
/*
  const auto d=m.named_modules("root");
  for(auto &i:d) {
   std::cerr << i.key() << "\n";
   std::cerr << *i.value() << "\n";
   for(auto& j:i.value()->named_parameters()) {
    std::cerr << j.key() << "\n";
   }
  }
  std::cerr << " ---------- top-level only -------------- \n";
*/
  std::string s;
  if(mname_(m).has_value()) s=mname(m);
 
  for(auto& p:m.named_parameters()) {
   std::cerr << s << "." << p.key() << "\n";
  }

  optgroup1(*o->o,m);
  
  return (K)0;
 KCATCH("optgroup");
}

KAPI optcheck(K x) {
 auto o1=torch::optim::AdagradOptions();
 auto o2=torch::optim::AdamOptions();
 auto o3=torch::optim::AdamWOptions();
 auto o4=torch::optim::LBFGSOptions();
 auto o5=torch::optim::RMSpropOptions();
 auto o6=torch::optim::SGDOptions(.01);
 double f=4.3333333333;
 bool b=x->g;
 auto v=b ? static_cast<int>(f) : static_cast<float>(f);
 std::cerr << "v: " << v << "\n";
 return K(0);
}


KAPI Xnone(K x,K y) {
 return kb(xnone(x,y->j));
}

J nest(K x) {
 if(x->t || !x->n) return 0;
 J n,m=0;
 for(J i=0;i<x->n;++i)
  if((n=nest(kK(x)[i])) && n>m) m=n;
 return ++m;
}

KAPI xnest(K x) {return kj(nest(x)); }

KAPI layerlist(K x) {
 Kmodule *q;
 if((q=xmodule(x))) {
  auto& m=mref(q->m);
  std::cerr << "named modules: \n";
  for(auto& a:m.named_modules())
   std::cerr << a.key() << "\n";
  std::cerr << "\nmodules: \n";
  for(auto& a:m.modules())
   std::cerr << *a << "\n";
  std::cerr << "\nparameters: \n";
  for(auto& a:m.named_parameters())
   std::cerr << a.key() << "\n";
  std::cerr << "\nbuffers: \n";
  for(auto& a:m.named_buffers())
   std::cerr << a.key() << "\n";
 }
 return (K)0;
}

#define OPTION(x,k,v) dictadd(x, lset(Setting::k), v)

/*
//#include  <c10/cuda/CUDAUtils.h>
KAPI dtest(K x) {
 auto d=c10::cuda::current_device();
 std::cerr << d << "\n";
 return(K)0;
}
*/

void xerror(const char* s,K x) {
 if(x->t) {
  AT_ERROR(s, kname(x));
 } else {
  switch(x->n) {
   case 0:  AT_ERROR(s, "empty list");
   case 1:  AT_ERROR(s, "1-element list containing ", kname(x));
   case 2:  AT_ERROR(s, "2-element list containing ", kname(kK(x)[0]), " and ", kname(kK(x)[1]));
   default: AT_ERROR(s, x->n, "-element list containing ", kname(kK(x)[0]), ", ", kname(kK(x)[1]),", ..");
  }
 }
}

KAPI knull(K x) {
 K r=ktn(0,0);
 std::cerr << "count of r: " << r->n << "\n";
 jk(&r,r1(x));
 std::cerr << "count of r: " << r->n << "\n";
 return r;
}
// void dictadd(K x, const char* s, K v){std::cerr << "dictadd\n"; K *k=kK(x); js(&k[0],cs(s)); std::cerr << "mid..";jk(&k[1],v); std::cerr << "dict exit\n";}

K findbuffer(K x,const std::string &s,short t=nh);
K findbuffer(K x,const std::string &s,short t) {
 TORCH_CHECK(xdict(x), "dictionary expected, ",kname(x)," given, unable to find parameter ",s);
 K k=kK(x)[0], v=kK(x)[1]; J i=kfind(k,s);
 if(i<0)
  return nullptr;
 TORCH_CHECK(!v->t, "general list of values expected, ",kname(v)," given, unable to find parameter ",s);
 K r=kK(v)[i];
 TORCH_CHECK(t==nh || t==r->t, s,": ",kname(t)," expected, ",kname(r->t)," supplied");
 return xnull(r) ? nullptr : r;
}

void putbuffers(K x,Cast c,const Tensor& t,const torch::optim::OptimizerParamState& p) {
 K v;
 switch(c) {
  case Cast::adagrad: {
   auto s=static_cast<const torch::optim::AdagradParamState&>(p);
   if((v=findbuffer(x,"step",-KJ))) s.step(v->j);
   if((v=findbuffer(x,"sum")))      s.sum(kput(v));
   break;
  }
  case Cast::adam: {
   auto s=static_cast<const torch::optim::AdamParamState&>(p);
   if((v=findbuffer(x,"step",-KJ)))       s.step(v->j);
   if((v=findbuffer(x,"exp_avg")))        s.exp_avg(kput(v));
   if((v=findbuffer(x,"exp_avg_sq")))     s.exp_avg_sq(kput(v));
   if((v=findbuffer(x,"max_exp_avg_sq"))) s.max_exp_avg_sq(kput(v));
   break;
  }
/*
  case Cast::lbfgs: {
   auto s=static_cast<const torch::optim::LBFGSParamState&>(p);
   dictadd(x, "func_evals",       kj(s.func_evals()));
   dictadd(x, "n_iter",           kj(s.n_iter()));
   dictadd(x, "t",                kf(s.t()));
   dictadd(x, "prev_loss",        kf(s.prev_loss()));
   dictadd(x, "d",                kget(s.d()));
   dictadd(x, "h_diag",           kget(s.H_diag()));
   dictadd(x, "prev_flag_grad",   kget(s.prev_flat_grad()));
   dictadd(x, "old_dirs",         kget(s.old_dirs()));
   dictadd(x, "old_stps",         kget(s.old_stps()));
   dictadd(x, "ro",               kget(s.ro()));
   dictadd(x, "al",               s.al() ? kget(s.al().value()) : ktn(0,0));
   break;
  }
  case Cast::rmsprop: {
   auto s=static_cast<const torch::optim::RMSpropParamState&>(p);
   dictadd(x, "step",       kj(s.step()));
   dictadd(x, "square_avg", kget(s.square_avg()));
   dictadd(x, "momentum",   kget(s.momentum_buffer()));
   dictadd(x, "grad_avg",   kget(s.grad_avg()));
   break;
  }
  case Cast::sgd: {
   auto s=static_cast<const torch::optim::SGDParamState&>(p);
   dictadd(x, "momentum",  kget(s.momentum_buffer()));
   break;
  }
*/
  default: AT_ERROR("unrecognized optimizer: ",(I)c,", unable to set parameter state");
 }
}

enum TensorPair {TensorPair};
enum TensorTriple {triple};
Tensor                           f1(const Tensor& x) {return x;}
std::tuple<Tensor,Tensor>        f1(const Tensor& x,enum TensorPair)   { return std::make_tuple(x,x+100);}
std::tuple<Tensor,Tensor,Tensor> f1(const Tensor& x,TensorTriple) { return std::make_tuple(x,x+10,x+1000);}

KAPI ftest1(K a) {
 Tensor x,y,z;
 x=torch::arange(8);
 switch(a->j) {
  case 2: std::tie(x,y)  =f1(x,TensorPair);   return kget(y);
  case 3: std::tie(x,y,z)=f1(x,triple); return kget(z);
  default: return kget(f1(x));
 }
}

using SeqResult=c10::variant<int,double,Tensor>;
SeqResult f(J a,Tensor x) { if(a) return x; else return 2.5;}

KAPI ftest2(K x) {
 KTRY
  Sequential q; SeqJoin j; Tensor a,b;
  j->push_back(Sequential());
  j->push_back(Sequential());
  j->push_back(AnyModule(Cat(0)));
  q->push_back(j);
  q->push_back(Reshape(std::vector<int64_t>{1,1,-1}));
  xtenpair(x,a,b);
  std::cerr << a << "\n";
  std::cerr << b << "\n";
  return kget(q->forward(a,b));
 KCATCH("ftest2");
}

KAPI ftest3(K x) {
 KTRY
  Sequential q;
  q->push_back(Reshape(std::vector<int64_t>{1,1,-1}));
  return kget(q->forward(kput(x)));
 KCATCH("ftest2");
}

auto childcount(const Module& m) {return m.children().size();}
auto childcount(Module* m) {std::cerr << "module ptr:\n"; return m->children().size();}
KAPI testcontainer(K x) {
 torch::nn::Linear m(1,2);
 std::cerr << childcount(*m) << "\n";
 std::cerr << childcount(m.get()) << "\n";
 return (K)0;
}

void testprint(int64_t d,const std::string s,const Module& m) {
  std::cerr << "  depth: " << d << ",";
  std::cerr << "   name: " << (s.size() ? s :  m.name());
  if(m.children().size()) {
   std::cerr << " " << m.name() << " (\n";
   for(auto& a:m.named_children())
    testprint(d+1,a.key(),*a.value());
   std::cerr << ")\n";
  } else {
  std::cerr << "\t" << m << "\n";
  }
}

/*
template <class... Fs> struct overload;

template <class F0, class... Frest> struct overload<F0, Frest...> : F0, overload<Frest...> {
    overload(F0 f0, Frest... rest) : F0(f0), overload<Frest...>(rest...) {}
    using F0::operator();
    using overload<Frest...>::operator();
};

template <class F0> struct overload<F0> : F0 {
    overload(F0 f0) : F0(f0) {}
    using F0::operator();
};

template <class... Fs> auto make_overload(Fs... fs) {
    return overload<Fs...>(fs...);
}
*/

//template <class... F> struct overload : F... {overload(F... f) : F(f)... {}};
//template <class... F> auto make_overload(F... f) {return overload<F...>(f...);}

//Tensor fwd(Layer& q,const Tensor& x,const Tensor& y) {c10::visit([&x,&y](auto& q) {q.ptr()->forward(x,y);}, q);}

std::tuple<Cast,K> mopt(bool,const Module&);

KAPI join1(K x) {
 KTRY
  SeqNest q;
  SeqJoin j;  // j=nullptr;
  q->push_back("xy", j);
  Sequential q1=Sequential(torch::nn::Embedding(10,50), torch::nn::Linear(50,784), Reshape(std::vector<int64_t>{-1,1,28,28}));
  j->push_back("zshape",q1);
  j->push_back("empty",Sequential());
  j->push_back("cat",AnyModule(Cat(1)));
  q->push_back("conv1",AnyModule(torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 64, 4).stride(2).padding(1).bias(false))));
  q->push_back("sig",AnyModule(torch::nn::Sigmoid()));
  q->push_back("flat",AnyModule(torch::nn::Flatten(torch::nn::FlattenOptions().start_dim(0).end_dim(-1))));
  return mget(true,true,*q);
 KCATCH("join1");
}

KAPI join2(K x) {
 KTRY
  SeqNest q;
  SeqJoin j;  // j=nullptr;
  q->push_back("xy", j);
  Sequential q1=Sequential(torch::nn::Embedding(10,50), torch::nn::Linear(50,784), Reshape(std::vector<int64_t>{-1,1,28,28}));
  j->push_back("zshape",q1);
  Sequential q2=Sequential(torch::nn::Identity());
  j->push_back("identity",q2);
  j->push_back("cat",AnyModule(Cat(1)));
  q->push_back("conv1",AnyModule(torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 64, 4).stride(2).padding(1).bias(false))));
  q->push_back("sig",AnyModule(torch::nn::Sigmoid()));
  q->push_back("flat",AnyModule(torch::nn::Flatten(torch::nn::FlattenOptions().start_dim(0).end_dim(-1))));
  return mget(true,true,*q);
 KCATCH("join2");
}

static K metrics(K x) {return (x->t==KS || x->t==-KS) ? x : nullptr;}

KAPI metrictest(K x) {
 KTRY
  TORCH_CHECK(!x->t && x->n>2, "bad arg(s)");
  K z=metrics(kK(x)[x->n-1]);
  if(z) {
   for(J i=0,n=z->t==KS ? z->n : 1; i<n; ++i) {
    S s=z->t==KS ? kS(z)[i] : z->s;
    //std::cerr << s << " -> " (I)metric(s) << "\n";
    std::cerr << s << "\n";
   }
  } else {
  }
  return (K)0;
 KCATCH("metrics");
}

KAPI ksizes(K x) {
 std::cerr << "type_info: " << sizeof(std::type_info) << "\n";
 J j; auto h=typeid(j).hash_code();
 std::cerr << "hash_code: " << sizeof(h) << "\n";
 std::cerr << "k0:      " << sizeof(k0) << "\n";
 std::cerr << "Tensor:  " << sizeof(Tensor) << "\n";
 std::cerr << "Module:  " << sizeof(Module) << "\n";
 std::cerr << "Layer:   " << sizeof(Layer) << "\n";
 std::cerr << "Class:   " << sizeof(Class) << "\n";
 std::cerr << "Cast:    " << sizeof(Cast) << "\n";
 std::cerr << "Ktag:    " << sizeof(Ktag) << "\n";
 std::cerr << "Kten:    " << sizeof(Kten) << "\n";
 std::cerr << "Kvec:    " << sizeof(Kvec) << "\n";
 std::cerr << "Kloss:   " << sizeof(Kloss) << "\n";
 std::cerr << "Kmodule: " << sizeof(Kmodule) << "\n";
 std::cerr << "Kopt:    " << sizeof(Kopt) << "\n";
 std::cerr << "Kmodel:  " << sizeof(Kmodel) << "\n";
 return (K)0;
}

KAPI randint_type(K x) {
 std::cerr << torch::randint(10,{3}) << "\n";
 return (K)0;
}

KAPI cudaloss(K x) {
 auto* l=xloss(x);
 auto& m=l->m;
 auto p=m.ptr();
 std::cerr << *p << "\n";
 p->to(torch::kCUDA);
 std::cerr << *p << "\n";
 if(l->c == Cast::nll) {
  auto g=m.get<torch::nn::NLLLoss>();
  std::cerr << g->options.weight() << "\n";
  std::cerr << g->weight << "\n";
  std::cerr << g->weight.is_same(g->options.weight()) << "\n";
 }
 return(K)0;
}

KAPI wt(K x,K y,K w) {
 KTRY
  torch::nn::BCELoss m;
  Tensor X,Y,W;
  if(xten(x,X) && xten(y,Y) && xten(w,W)) {
   m->options.weight(W);
   Tensor r=m->forward(X,Y);
   m->options.weight({});
   return kget(r);
  } else {
   return(K)0;
  }
 KCATCH("bce wt");
}

KAPI memtest(K x) {
 std::vector<torch::Tensor> v;
 v.reserve(10000000);
 for(size_t i=0; i<10000000; ++i) v.emplace_back(torch::arange(10));
 return kj(v.size());
}

//#include <c10/cuda/CUDAMacros.h>
//#include <c10/cuda/CUDACachingAllocator.h>

// check for cuda via USE_CUDA
// #ifdef USE_CUDA
//  ..
// #endif
/*
namespace c10 {
namespace cuda {
namespace CUDACachingAllocator {
C10_CUDA_API void emptyCache();
C10_CUDA_API uint64_t currentMemoryAllocated(int device);
C10_CUDA_API uint64_t maxMemoryAllocated(int device);
C10_CUDA_API void     resetMaxMemoryAllocated(int device);
C10_CUDA_API uint64_t currentMemoryCached(int device);
C10_CUDA_API uint64_t maxMemoryCached(int device);
C10_CUDA_API void     resetMaxMemoryCached(int device);
}}}
*/

/*
cache      
memory     e.g. memory() or memory`cuda or memory 0
maxcache   
maxmemory  
emptycache
resetcache 
resetmemory
*/
KAPI cudamem(K x) {
 KTRY
  // if sym, get device no
  // if int, verify -1<n< env.cuda
  //auto n=c10::cuda::CUDACachingAllocator::currentMemoryAllocated(x->j);
  //return kj(n);
  return kj(nj);
 KCATCH("cuda memory");
}

using Reduce1=c10::variant<torch::enumtype::kNone, torch::enumtype::kMean, torch::enumtype::kSum>;
using Reduce2=c10::variant<torch::enumtype::kNone, torch::enumtype::kBatchMean, torch::enumtype::kSum, torch::enumtype::kMean>;

static void reduce(Reduce1& r,Cast c,S s) {
 switch(emap(s)) {
  case Enum::none: r=torch::kNone; break;
  case Enum::mean: r=torch::kMean; break;
  case Enum::sum:  r=torch::kSum; break;
  default: AT_ERROR("not one of none,mean,sum");
 }
}

static void reduce(Reduce2& r,Cast c,S s) {
 switch(emap(s)) {
  case Enum::none:      r=torch::kNone; break;
  case Enum::batchmean: r=torch::kBatchMean; break;
  case Enum::mean:      r=torch::kMean; break;
  case Enum::sum:       r=torch::kSum; break;
  default: AT_ERROR("not one of none,batchmean,mean,sum");
 }
}

KAPI losstest(K x) {
 KTRY
 torch::nn::L1LossOptions o1;
 torch::nn::KLDivLossOptions o2;
 reduce(o1.reduction(),Cast::l1,x->s);
 std::cerr << torch::enumtype::get_enum_name(o1.reduction()) << "\n";
 reduce(o2.reduction(),Cast::kl,x->s);
 std::cerr << torch::enumtype::get_enum_name(o2.reduction()) << "\n";
 return (K)0;
 KCATCH("loss reduction");
}

#define ENUMTEST(name) \
{ \
  v = torch::k##name; \
  std::cerr << torch::enumtype::get_enum_name(v) << " " << ESYM(v) << "\n"; \
}

KAPI enumtest(K x) {
  c10::variant<
    torch::enumtype::kLinear,
    torch::enumtype::kConv1D,
    torch::enumtype::kConv2D,
    torch::enumtype::kConv3D,
    torch::enumtype::kConvTranspose1D,
    torch::enumtype::kConvTranspose2D,
    torch::enumtype::kConvTranspose3D,
    torch::enumtype::kSigmoid,
    torch::enumtype::kTanh,
    torch::enumtype::kReLU,
    torch::enumtype::kLeakyReLU,
    torch::enumtype::kFanIn,
    torch::enumtype::kFanOut,
    torch::enumtype::kConstant,
    torch::enumtype::kReflect,
    torch::enumtype::kReplicate,
    torch::enumtype::kCircular,
    torch::enumtype::kNearest,
    torch::enumtype::kBilinear,
    torch::enumtype::kBicubic,
    torch::enumtype::kTrilinear,
    torch::enumtype::kArea,
    torch::enumtype::kSum,
    torch::enumtype::kMean,
    torch::enumtype::kMax,
    torch::enumtype::kNone,
    torch::enumtype::kBatchMean,
    torch::enumtype::kZeros,
    torch::enumtype::kBorder,
    torch::enumtype::kReflection
  > v;

  ENUMTEST(Linear)
  ENUMTEST(Conv1D)
  ENUMTEST(Conv2D)
  ENUMTEST(Conv3D)
  ENUMTEST(ConvTranspose1D)
  ENUMTEST(ConvTranspose2D)
  ENUMTEST(ConvTranspose3D)
  ENUMTEST(Sigmoid)
  ENUMTEST(Tanh)
  ENUMTEST(ReLU)
  ENUMTEST(LeakyReLU)
  ENUMTEST(FanIn)
  ENUMTEST(FanOut)
  ENUMTEST(Constant)
  ENUMTEST(Reflect)
  ENUMTEST(Replicate)
  ENUMTEST(Circular)
  ENUMTEST(Nearest)
  ENUMTEST(Bilinear)
  ENUMTEST(Bicubic)
  ENUMTEST(Trilinear)
  ENUMTEST(Area)
  ENUMTEST(Sum)
  ENUMTEST(Mean)
  ENUMTEST(Max)
  ENUMTEST(None)
  ENUMTEST(BatchMean)
  ENUMTEST(Zeros)
  ENUMTEST(Border)
  ENUMTEST(Reflection)
 return (K)0;
}

KAPI kdata(K x,K y) {
 KTRY
  int64_t i=0;
  auto dataset = torch::data::datasets::MNIST(x->s)
    .map(torch::data::transforms::Normalize<>(0.5, 0.5))
    .map(torch::data::transforms::Stack<>());
  auto data_loader = torch::data::make_data_loader(std::move(dataset));
  for (torch::data::Example<>& batch : *data_loader) {
    if(i==y->j) {
     std::cout << batch.target << "\n";
     std::cout << batch.data   << "\n ";
     return kten(batch.data);
    }
  }
  return (K)0;
 KCATCH("mnist test");
}

void f(int64_t n) {
 n*=1000000;
 auto t=torch::rand(n);
 double d=0; float f=0,*p=t.data_ptr<float>();
 for(int64_t i=0; i<n; ++i) d+=p[i], f+=p[i];
 std::cerr << "double  sum: " <<   d << "\n";
 std::cerr << "float   sum: " <<   f << "\n\n";
 std::cerr << "double mean: " << d/n << "\n";
 std::cerr << "float  mean: " << f/n << "\n";
 std::cerr << "torch  mean: " << t.mean().item().toFloat() << "\n";
}

KAPI ftest(K x) {
 f(x->j);
 return (K)0;
}

KAPI hashtest(K x) {
 std::unordered_set<J> Ptrs;
 Ptrs.insert(10);
 Ptrs.insert(20);
 Ptrs.insert(2);
 Ptrs.insert(20);

 std::cerr << "size: " << Ptrs.size() << "\n";
 for(auto j:Ptrs)
  std::cerr << j << "\n";

 //std::cerr << " find: " << Ptrs.find(20) << "\n";
 std::cerr << "count: " << Ptrs.count(20) << "\n";
 Ptrs.erase(20);
 std::cerr << "size: " << Ptrs.size() << "\n";
 //std::cerr << " find: " << Ptrs.find(20) << "\n";
 std::cerr << "count: " << Ptrs.count(20) << "\n";
 return (K)0;
}

void errfail() {
 if(true) {
  AT_ERROR("err");
 } else {
  AT_ERROR("false");
 }
}

KAPI testcount(K x,K y) {
 KTRY
 if(y->t != -KJ) return KERR("2nd arg must be offset");
 Pairs p; J i=y->j; J n=xargc(x,i,p);
 std::cerr << "arg count: " << n << ", pair count: " << p.n << "\n";
 return kb(xnone(x,i));
 KCATCH("test count");
}

KAPI testptr(K x) {
 Tensor t;
 if(xten(x,t))
  std::cerr<<"tensor\n";
 else if(xten(x,0,t) && x->n==1)
  std::cerr<<"enlisted tensor\n";
 else
  std::cerr<<"something else\n";
 return(K)0;
}

#define ASSERT_THROWS_WITH(statement, substring)                        \
  {                                                                     \
    std::string assert_throws_with_error_message;                       \
    try {                                                               \
      (void)statement;                                                  \
      std::cerr << "Expected statement `" #statement                       \
                "` to throw an exception, but it did not";              \
    } catch (const c10::Error& e) {                                     \
      assert_throws_with_error_message = e.what_without_backtrace();    \
    } catch (const std::exception& e) {                                 \
      assert_throws_with_error_message = e.what();                      \
    }                                                                   \
    if (assert_throws_with_error_message.find(substring) ==             \
        std::string::npos) {                                            \
      std::cerr << "Error message \"" << assert_throws_with_error_message  \
             << "\" did not contain expected substring \"" << substring \
             << "\"";                                                   \
    }                                                                   \
  }

KAPI namecheck(K x) {
 torch::nn::Sequential s; //size_t n=0;
 std::cout << "initial size: " << s->size() << "\n";
 ASSERT_THROWS_WITH(
      s->push_back("name.with.dot", torch::nn::Linear(3, 4)),
      "Submodule name must not contain a dot (got 'name.with.dot')");
  ASSERT_THROWS_WITH(
      s->push_back("", torch::nn::Linear(3, 4)),
      "Submodule name must not be empty");
  std::cout << "size after name errors: " << s->size() << "\n";
  //for(auto&c:s->named_children()) n++;
  std::cout << "size of modules: "        << s->modules(false).size() << "\n";
  std::cout << "size of named children: " << s->named_children().size() << "\n";

  return(K)0;
}

KAPI dupname(K x) {
KTRY
 Sequential q(
  {{"A", torch::nn::Linear(1,2)},
   {"B", torch::nn::Conv2d(3,4,5)}});
 return (K)0;
KCATCH("duplicate names");
}

KAPI kdictflag(K x) {
 return kb(xdict(x));
}

static K ksub(K x,const char* e) {
 KTRY
 std::cerr << "in ksub " << (!x ? "null" : "with args")<< "\n";
 Tensor t;
 if(!x || (x->t==-KS && x->s==cs("help"))) {
  std::cerr << " still in ksub " << (!x ? "null" : "with args")<< "\n";
  AT_ERROR(e," help here..");
 }

 if(xten(x,t)) {
  return kten(torch::max(t));
 } else {
  return ksub(nullptr,e);
 }
 KCATCH(e);
}

KAPI ktest(K x) {return ksub(x,"ktest()");}

KAPI mixtest(K x) {return kb(xmixed(x,4));}

K help(bool b,const char* s) {return b ? KERR(s) : (fprintf(stderr,"%s\n",s), (K)0);}

#define KHELP(cond, ...)  \
  if((cond))              \
   AT_WARN(__VA_ARGS__);  \
  else                    \
   AT_ERROR(__VA_ARGS__); \

typedef struct {
 std::array<std::tuple<S,Cast,std::function<Tensor(Tensor)>>,2> fn = {{
 }};
} Testenv;

Testenv& testenv() {static Testenv e; return e;}

KAPI pairtest(K x) {
 KTRY
 Pairs p;
 if(xpairs(x,p) || xpairs(x,x->n-1,p)) {
  switch(p.a) {
   case 1: std::cout << "dictionary["; break;
   case 2: std::cout << "pairs["; break;
   case 3: std::cout << "list["; break;
   case 4: std::cout << "symbol list["; break;
   default: std::cout << "unknown name,value structure["; break;
  }
  std::cout << p.n << "]\n";
  while(xpair(p)) {
   switch(p.t) {
    case -KB: std::cout << "boolean: " << p.k << " -> " << p.b << "\n"; break;
    case -KS: std::cout << " symbol: " << p.k << " -> " << p.s << "\n"; break;
    case -KJ: std::cout << "integer: " << p.k << " -> " << p.j << "\n"; break;
    case -KF: std::cout << " double: " << p.k << " -> " << p.f << "\n"; break;
    default:  std::cout << "  other: " << p.k << " -> " << kname(p.t) << "\n"; break;
   }
  }
  return kb(true);
 } else {
  return kb(false);
 }
 KCATCH("pairs test..");
}

KAPI lbfgs(K x) {
    int i, n=x->j;
    auto t=torch::randn({n});

    TensorVector v = {torch::randn({n}, torch::requires_grad())};
    //torch::optim::SGD o(v, /*lr=*/0.01);
    torch::optim::LBFGS o(v, 1);

    auto cost = [&](){
        o.zero_grad();
        auto d = torch::pow(v[0] - t, 2).sum();
        std::cerr << i << ") " << d.item().toDouble() << "\n";
        d.backward();
        return d;
    };

    for (i = 0; i < 5; ++i){
        o.step(cost);//for LBFGS
        //cost(); o.step(); // for SGD
    }
    return kget(torch::stack({t,v[0]}));
}

KAPI learn(K x) {
 KTRY
  Scalar s; Tensor t;
  if(xten(x,0,t) && xnum(x,1,s)) {
   if(t.grad().defined()) {
    torch::NoGradGuard g;
    //t.add_(-s.toDouble()*t.grad());
    t.add_(-s*t.grad());
    t.grad().zero_();
    return (K)0;
   } else {
    return KERR("no gradient defined");
   }
  } else {
   return KERR("unrecognized arg(s), expecting (tensor;learning rate)");
  }
 KCATCH("error applying learning rate and gradient to tensor");
}

bool xindex(K x,Tensor &t) {J n,*j; return xten(x,t) ? true : (xlong(x,n,j) ? t=kput(x),true : false);}
bool xindex(K x,J i,Tensor &t) { return xind(x,i) && xindex(kK(x)[i],t);}

KAPI kindex(K x) {
 J d; Tensor r,t,i;
 KTRY
  if(xten(x,0,t) && xlong(x,1,d) && xindex(x,2,i)) {
    if(x->n==3)
     return kten(torch::index_select(t,d,i));
    else if(xten(x,3,r) && x->n==4)
     return torch::index_select_out(r,t,d,i), (K)0;
  }
  return KERR("unrecognized arg(s), expected (tensor;dim;indices;optional output tensor)");
 KCATCH("index");
}

KAPI opttest(K x) {
 TensorOptions o;
 //o.is_variable(true);
 std::cout << "dtype:       " << o.dtype() << "\n";
 std::cout << "device:      " << o.device() << "\n";
 std::cout << "layout:      " << o.layout() << "\n";
 std::cout << "gradient:    " << o.requires_grad() << "\n";
 //std::cout << "variable:    " << o.is_variable() << "\n";
 std::cout << "has dtype:   " << o.has_dtype()  << "\n";
 std::cout << "has device:  " << o.has_device() << "\n";
 std::cout << "has layout:  " << o.has_layout() << "\n";
 //std::cout << "has variable:" << o.has_is_variable() << "\n";
 std::cout << "has gradient:" << o.has_requires_grad() << "\n";
 return (K)0;
}

// tensor(`sparse;array)
// tensor(`sparse;array;mask)
// tensor(`sparse;size) -> tensor(`empty;3 4;`sparse)

/*
KAPI sparse(K x) {
 Tensor a,b; TensorOptions o;
 KTRY
  if(xten(x,a)) {
  } else if(xten(x,0,a) && xten(x,1,b) {
   if(x->n==2)
   else if(x->n==3)
   
    // no size
    // size
    // w'options
  } else if(x->n = {
  }
 
  return(K)0;
 KCATCH("sparse tensor error");
}
*/

KAPI to_sparse(K x) {
 if(auto* t=xten(x))
  return kten(t->to_sparse());
 else
  AT_ERROR("to_sparse not implemented for ",kname(x->t));
}


KAPI sparse1(K x) {
 auto m=kput(kK(x)[0]),t=kput(kK(x)[1]),v=torch::masked_select(t,m),i=torch::nonzero(m);
 //return kten(torch::sparse_coo_tensor(i.t(),v));
 return kten(torch::sparse_coo_tensor(i.t(),v,m.sizes()));
}
