#include "ktorch.h"

namespace nn=torch::nn;

KAPI trainx(K x) {
 KTRY
  TORCH_CHECK(!x->t, "train: not implemented for ",kname(x));
  TORCH_CHECK(x->n>2, "train: at least 3 args expected, given ",x->n);
  Kmodel *m=xmodel(x,0);
  TORCH_CHECK(m, "train: 1st arg of model expected, given ",kname(x,0));
  if(auto *v=xvec(x,1)) {
   // (m;v;w)
   // (m;v;ix;iy;w)
  } else if(auto *d=xtensordict(x,1)) {
   // (m;d;kx;ky;w)
   // (m;d;w)
  } else if(xten(x,1)) {
   // (m;x;y;w)
  } else {
   AT_ERROR("train: unrecognized 2nd arg, expecting tensor, vector or dictionary of tensor, given ",kname(x,1));
  }
  return (K)0;
 KCATCH("train");
}

KAPI names1(K x) {
 KTRY
  Kmodule *k=xmodule(x);
  TORCH_CHECK(k, "need module");
  const auto& v=(*k->m).named_modules().keys();
  J i=0; K s=ktn(KS,v.size());
  for(const auto& a:v)
   kS(s)[i++]=cs(a.c_str());
  return s;
 KCATCH("names");
}

KAPI parms(K x) {
 KTRY
  K y=nullptr; auto *g=xtag(x); Moduleptr m;
  if(!g) {
   g=xtag(x,0);
   TORCH_CHECK(g, "parms: expecting module, model or optimizer, not implemented for ",kname(x));
   TORCH_CHECK(!x->t && x->n==2, "parms: expecting up to two args, (object; index/name), given ",x->n," args");
   y=kK(x)[1];
   TORCH_CHECK(y->t==-KJ || y->t==-KS, "parms: expecting integer or symbol for 2nd arg, given ",kname(x,1));
  }
  switch(g->a) {
   case Class::loss:
   case Class::module:    m=((Kmodule*)g)->m; break;
   case Class::model:     m=((Kmodel*) g)->m; break;
   case Class::optimizer: m=((Kopt*)   g)->m; break;
   default: AT_ERROR("parms: not implemented for ",mapclass(g->a));
  }
  S s=mname(*m);
  if(y) {
   if(y->t == -KJ) {
    const auto& v=m->modules(false); J n=v.size();
    TORCH_CHECK(y->j < n, "parms: invalid index[",y->j,"] for ",n," module",(n==1 ? "" : "s"));
    if(-1<y->j) m=v.at(y->j);
   } else {
    m=m->named_modules(s ? s : "")[y->s];
   }
  }
  return kdict(m->named_parameters());
 KCATCH("parms/buffers");
}

KAPI names(K x) {
 KTRY
  auto *g=xtag(x); Moduleptr m;
  switch(g->a) {
   case Class::loss:
   case Class::module:    m=((Kmodule*)g)->m; break;
   case Class::model:     m=((Kmodel*) g)->m; break;
   case Class::optimizer: m=((Kopt*)   g)->m; break;
   default: AT_ERROR("parms: not implemented for ",mapclass(g->a));
  }
  S s=mname(*m);
  const auto& k=m->named_modules(s ? s : "").keys();
  J i=0; K z=ktn(KS,k.size());
  for(const auto& a:k) kS(z)[i++]=cs(a.c_str());
  return z;
 KCATCH("names");
}

static void dups() {
 torch::nn::Linear m(1,1);
 torch::optim::SGD o(m->parameters(),.1);
 torch::optim::OptimizerParamGroup g(m->parameters());
 o.add_param_group(g);
 auto& p=o.param_groups();
 if(p[0].params()[1].is_same(p[1].params()[1]))
  std::cerr << "same parameter across different groups\n";
}

KAPI duptest(K x) {
 KTRY
 dups();
 return (K)0;
 KCATCH("dups");
}
 
void mdict2(const Moduleptr& m) {
 if(m)
  std::cerr << "ptr holds something: " << *m << "\n";
 else
  std::cerr << "ptr is empty..\n";
}

void mdict1() {
 Moduleptr p;
 mdict2(p);
 nn::ParameterDict dict;
 torch::Tensor ta = torch::randn({1, 2}, torch::requires_grad(true));
 torch::Tensor tb = torch::randn({1, 2}, torch::requires_grad(false));
 dict->insert("A", ta);
 dict->insert("B", tb);
 p=dict.ptr();
 mdict2(p);
 nn::Linear l(2,3);
 nn::Sequential q(l);
 nn::ModuleDict m; m->update({{"seq",q.ptr()}}); m->update({{"parms",dict.ptr()}});
 //nn::ModuleList m(q);
 m->register_parameter("test",torch::randn(4));
 std::cerr << m << "\n";
 std::cerr << "m->named_parameters(true)()\n";
 for(const auto& a:m->named_parameters(true)) std::cerr << a.key() << "\n";
 std::cerr << "m->named_modules()\n";
 for(const auto& a:m->named_modules())
  std::cerr << a.key() << "\n";
 //mdict2(m.ptr());
}

KAPI mdict(K x) {
 mdict1();
 return (K)0;
}
 
/*
  AnyModule(std::shared_ptr<ModuleType> module)
  AnyModule(ModuleType&& module)
  AnyModule(const ModuleHolder<ModuleType>& module_holder)
*/
 
static void checkhash(const Moduleptr& x) {
 Cast c=mcast(*x);
 std::cerr << "Cast: " << (I)c << ", " << msym(*x) << "\n";
}

static void addmodule(nn::Sequential& x,const Moduleptr& y) {
 x->push_back(*y->as<torch::nn::Linear>());
}

static void addmodule(const Moduleptr& x,const Moduleptr& y) {
 x->as<nn::Sequential>()->push_back(*y->as<torch::nn::Linear>());
}

static void checkptr(const Moduleptr& x,const Moduleptr& y) {
 if(auto a=std::dynamic_pointer_cast<nn::SequentialImpl>(x)) {
  std::cerr << "1st arg is a Sequential module\n";
 }
 if(auto a=std::dynamic_pointer_cast<nn::LinearImpl>(y)) {
  std::cerr << "2nd arg is a Linear module\n";
  nn::Linear l(a);
 }
}

KAPI cast(K x) {
 KTRY
  torch::nn::Sequential a;
  torch::nn::Linear l(1,2);
  auto q=a.ptr();
  auto p=l.ptr();
  checkptr(q,p);
  checkhash(q);
  checkhash(p);
  addmodule(q,p);
  addmodule(a,p);
  a->push_back(p);
  q->push_back(p);
  std::cerr << *q << "\n";
  return (K)0;
 KCATCH("cast");
}

Tensor c1(Cast c,Moduleptr& m,const Tensor& t) {
 switch(c) {
  case Cast::adaptavg1d:      AT_ERROR("nyi");
  case Cast::adaptavg2d:      AT_ERROR("nyi");
  case Cast::adaptavg3d:      AT_ERROR("nyi");
  case Cast::adaptmax1d:      AT_ERROR("nyi");
  case Cast::adaptmax2d:      AT_ERROR("nyi");
  case Cast::adaptmax3d:      AT_ERROR("nyi");
  case Cast::adrop:           AT_ERROR("nyi");
  case Cast::attention:       AT_ERROR("nyi");
  case Cast::avgpool1d:       AT_ERROR("nyi");
  case Cast::avgpool2d:       AT_ERROR("nyi");
  case Cast::avgpool3d:       AT_ERROR("nyi");
  case Cast::base:            AT_ERROR("nyi");
  case Cast::batchnorm1d:     AT_ERROR("nyi");
  case Cast::batchnorm2d:     AT_ERROR("nyi");
  case Cast::batchnorm3d:     AT_ERROR("nyi");
  case Cast::bilinear:        AT_ERROR("nyi");
  case Cast::cat:             AT_ERROR("nyi");
  case Cast::celu:            AT_ERROR("nyi");
  case Cast::conv1d:          AT_ERROR("nyi");
  case Cast::conv2d:          AT_ERROR("nyi");
  case Cast::conv3d:          AT_ERROR("nyi");
  case Cast::convtranspose1d: AT_ERROR("nyi");
  case Cast::convtranspose2d: AT_ERROR("nyi");
  case Cast::convtranspose3d: AT_ERROR("nyi");
  case Cast::crossmap2d:      AT_ERROR("nyi");
  case Cast::decoder:         AT_ERROR("nyi");
  case Cast::decoderlayer:    AT_ERROR("nyi");
  case Cast::drop:            AT_ERROR("nyi");
  case Cast::drop2d:          AT_ERROR("nyi");
  case Cast::drop3d:          AT_ERROR("nyi");
  case Cast::elu:             AT_ERROR("nyi");
  case Cast::embed:           AT_ERROR("nyi");
  case Cast::embedbag:        AT_ERROR("nyi");
  case Cast::encoder:         AT_ERROR("nyi");
  case Cast::encoderlayer:    AT_ERROR("nyi");
  case Cast::expand:          AT_ERROR("nyi");
  case Cast::fadrop:          AT_ERROR("nyi");
  case Cast::flatten:         AT_ERROR("nyi");
  case Cast::fmaxpool2d:      AT_ERROR("nyi");
  case Cast::fmaxpool3d:      AT_ERROR("nyi");
  case Cast::fold:            AT_ERROR("nyi");
  case Cast::gelu:            AT_ERROR("nyi");
  case Cast::glu:             AT_ERROR("nyi");
  case Cast::groupnorm:       AT_ERROR("nyi");
  case Cast::gru:             AT_ERROR("nyi");
  case Cast::hardshrink:      AT_ERROR("nyi");
  case Cast::hardtanh:        AT_ERROR("nyi");
  case Cast::identity:        AT_ERROR("nyi");
  case Cast::instancenorm1d:  AT_ERROR("nyi");
  case Cast::instancenorm2d:  AT_ERROR("nyi");
  case Cast::instancenorm3d:  AT_ERROR("nyi");
  case Cast::interpolate:     AT_ERROR("nyi");
  case Cast::layernorm:       AT_ERROR("nyi");
  case Cast::leakyrelu:       AT_ERROR("nyi");
  case Cast::linear:          return std::dynamic_pointer_cast<torch::nn::LinearImpl>(m)->forward(t);
  case Cast::localnorm:       AT_ERROR("nyi");
  case Cast::logsigmoid:      AT_ERROR("nyi");
  case Cast::logsoftmax:      AT_ERROR("nyi");
  case Cast::lppool1d:        AT_ERROR("nyi");
  case Cast::lppool2d:        AT_ERROR("nyi");
  case Cast::lstm:            AT_ERROR("nyi");
  case Cast::maxpool1d:       AT_ERROR("nyi");
  case Cast::maxpool2d:       AT_ERROR("nyi");
  case Cast::maxpool3d:       AT_ERROR("nyi");
  case Cast::modulelist:      AT_ERROR("nyi");
  case Cast::mul:             AT_ERROR("nyi");
  case Cast::normalize:       AT_ERROR("nyi");
  case Cast::pad:             AT_ERROR("nyi");
  case Cast::pad1d:           AT_ERROR("nyi");
  case Cast::pad2d:           AT_ERROR("nyi");
  case Cast::pad3d:           AT_ERROR("nyi");
  case Cast::pairwise:        AT_ERROR("nyi");
  case Cast::prelu:           AT_ERROR("nyi");
  case Cast::reflect1d:       AT_ERROR("nyi");
  case Cast::reflect2d:       AT_ERROR("nyi");
  case Cast::relu:            AT_ERROR("nyi");
  case Cast::relu6:           AT_ERROR("nyi");
  case Cast::replicate1d:     AT_ERROR("nyi");
  case Cast::replicate2d:     AT_ERROR("nyi");
  case Cast::replicate3d:     AT_ERROR("nyi");
  case Cast::reshape:         AT_ERROR("nyi");
  case Cast::rnn:             AT_ERROR("nyi");
  case Cast::rrelu:           AT_ERROR("nyi");
  case Cast::selu:            AT_ERROR("nyi");
  case Cast::seqjoin:         AT_ERROR("nyi");
  case Cast::seqnest:         AT_ERROR("nyi");
  case Cast::sequential:      AT_ERROR("nyi");
  case Cast::sigmoid:         AT_ERROR("nyi");
  case Cast::similar:         AT_ERROR("nyi");
  case Cast::softmax:         AT_ERROR("nyi");
  case Cast::softmax2d:       AT_ERROR("nyi");
  case Cast::softmin:         AT_ERROR("nyi");
  case Cast::softplus:        AT_ERROR("nyi");
  case Cast::softshrink:      AT_ERROR("nyi");
  case Cast::softsign:        AT_ERROR("nyi");
  case Cast::squeeze:         AT_ERROR("nyi");
  case Cast::tanh:            AT_ERROR("nyi");
  case Cast::tanhshrink:      AT_ERROR("nyi");
  case Cast::threshold:       AT_ERROR("nyi");
  case Cast::transformer:     AT_ERROR("nyi");
  case Cast::unfold:          AT_ERROR("nyi");
  case Cast::unsqueeze:       AT_ERROR("nyi");
  case Cast::upsample:        AT_ERROR("nyi");
  case Cast::zeropad2d:       AT_ERROR("nyi");
  default: AT_ERROR("unrecognized module");
 }
}

void f1() {
 auto t=torch::randn({100,200});
 torch::nn::Linear l(200,100);
 for(size_t i=0; i<1000000; ++i)
  auto r=l->forward(t);
}

void f2(Moduleptr m) {
 auto t=torch::randn({100,200});
 for(size_t i=0; i<1000000; ++i)
  //auto r=std::dynamic_pointer_cast<torch::nn::LinearImpl>(m)->forward(t);
  auto r=c1(Cast::linear, m, t);
}

void f3(AnyModule& a) {
 auto t=torch::randn({100,200});
 for(size_t i=0; i<1000000; ++i)
  auto r=a.forward(t);
}


KAPI a(K x) {
 KTRY
  AnyModule a(torch::nn::Linear(200,100));
  f3(a);
  return (K)0;
 KCATCH("timer");
}

KAPI f(K x) {
 KTRY
  //f1();
  BaseModule m;
  m->register_module("linear",torch::nn::Linear(200,100));
  m->register_parameter("tensor",torch::randn(10));
  f2(m->children()[0]);
  return (K)0;
 KCATCH("timer");
}

KAPI g(K x) {
 KTRY
  f1();
  return (K)0;
 KCATCH("timer");
}

KAPI f_old(K x) {
 KTRY
  BaseModule m;
  m->register_module("linear",torch::nn::Linear(1,2));
  m->register_parameter("tensor",torch::randn(10));
  auto a=AnyModule(m);
  std::cerr << *m << "\n";
  //return kdict(m->named_parameters());
  //return mget(true,true,*m);
  //return mget(true,true,*m->children()[0]);
  //return mget(true,true,*a.ptr());
  // static cast won't compile
  /* fatal error: cannot cast 'torch::nn::Module *' to
      'typename _Sp::element_type *' (aka 'torch::nn::LinearImpl *') via virtual base 'torch::nn::Module'
      'std::static_pointer_cast<torch::nn::LinearImpl, torch::nn::Module>'
  */
  //AnyModule l(std::dynamic_pointer_cast<torch::nn::LinearImpl>(m->children()[0]));
  //auto p=std::dynamic_pointer_cast<torch::nn::SigmoidImpl>(m->children()[0]);
  auto p=std::dynamic_pointer_cast<torch::nn::LinearImpl>(m->children()[0]);
  if(p) {
   return mget(true,true,*p);
  } else {
   return kb(false);
  }
 KCATCH("test base module")
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

J nest(K x) {
 if(x->t || !x->n) return 0;
 J n,m=0;
 for(J i=0;i<x->n;++i)
  if((n=nest(kK(x)[i])) && n>m) m=n;
 return ++m;
}

KAPI xnest(K x) {return kj(nest(x)); }

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
  nn::Sequential q; SeqJoin j; Tensor a,b;
  j->push_back(nn::Sequential());
  j->push_back(nn::Sequential());
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
  nn::Sequential q;
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

KAPI join1(K x) {
 KTRY
  SeqNest q;
  SeqJoin j;  // j=nullptr;
  q->push_back("xy", j);
  nn::Sequential q1=nn::Sequential(torch::nn::Embedding(10,50), torch::nn::Linear(50,784), Reshape(std::vector<int64_t>{-1,1,28,28}));
  j->push_back("zshape",q1);
  j->push_back("empty",nn::Sequential());
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
  nn::Sequential q1=nn::Sequential(torch::nn::Embedding(10,50), torch::nn::Linear(50,784), Reshape(std::vector<int64_t>{-1,1,28,28}));
  j->push_back("zshape",q1);
  nn::Sequential q2=nn::Sequential(torch::nn::Identity());
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
 std::cerr << "Class:   " << sizeof(Class) << "\n";
 std::cerr << "Cast:    " << sizeof(Cast) << "\n";
 std::cerr << "Ktag:    " << sizeof(Ktag) << "\n";
 std::cerr << "Kten:    " << sizeof(Kten) << "\n";
 std::cerr << "Kvec:    " << sizeof(Kvec) << "\n";
 std::cerr << "Kmodule: " << sizeof(Kmodule) << "\n";
 std::cerr << "Kopt:    " << sizeof(Kopt) << "\n";
 std::cerr << "Kmodel:  " << sizeof(Kmodel) << "\n";
 return (K)0;
}

KAPI randint_type(K x) {
 std::cerr << torch::randint(10,{3}) << "\n";
 return (K)0;
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
 nn::Sequential q(
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
