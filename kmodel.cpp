#include "ktorch.h"
#define OPTION(x,k,v) dictadd(x, modelopt(Setting::k), v)

// ---------------------------------------------------------------------------------------
// resetgrad - zero out or set gradient to none if boolean flag set true
// zerograd - zero gradients on tensor, vector of tensors, optimizer, module or model
// nograd - set gradient to none for tensor, vector of tensors, optimizer, module or model
// ---------------------------------------------------------------------------------------
static void resetgrad(const Tensor& t,bool b){
 auto &g=t.mutable_grad();
 if(g.defined()) {
  g=g.detach();
  if(b)
   g.reset();
  else
   g.zero_();
 }
}

static void resetgrad(const TensorVector& v,bool b){
 for(const auto& t:v)
  resetgrad(t,b);
}

static void resetgrad(const TensorDict& d,bool b){
 for(const auto& i:d)
  resetgrad(i.value(),b);
}

static void resetgrad(Module& m,bool b){
 m.zero_grad(b);
}

static void resetgrad(Optimizer& o,bool b){
 if(b) {
  for(const auto& p:o.param_groups())
   for(const auto &t:p.params())
    resetgrad(t,b);
 } else {
  o.zero_grad();
 }
}

K resetgrad(K x,bool b,const char *c) {
 KTRY
  auto *g=xtag(x);
  TORCH_CHECK(g, c,": not implemented for ",kname(x));
  switch(g->a) {
   case Class::tensor:     resetgrad(g->tensor(),b); break;
   case Class::vector:     resetgrad(g->vector(),b); break;
   case Class::dict:       resetgrad(g->dict(),b); break;
   case Class::module:     resetgrad(g->module(),b); break;
   case Class::optimizer:
   case Class::model:      resetgrad(g->opt(),b); break;
   default: TORCH_ERROR(c,": not implemented for ",mapclass(g->a));
  }
  return (K)0;
 KCATCH(c);
}

KAPI   nograd(K x) {return resetgrad(x, true,  "nograd");}
KAPI zerograd(K x) {return resetgrad(x, false, "zerograd");}

// -------------------------------------------------------------------
//  metric  - map k symbol <-> metric, e.g. `output -> Metric::output
// -------------------------------------------------------------------
static Metric metric(S s) {
 for(const auto& m:env().metric)
  if(std::get<0>(m)==s) return std::get<1>(m);
 TORCH_ERROR("unrecognized metric: ",s);
}

static S metric(Metric m) {
 for(const auto& a:env().metric)
  if(std::get<1>(a)==m) return std::get<0>(a);
 TORCH_ERROR("unrecognized metric: ",(I)m);
}

// -------------------------------------------------------------------------------------------
// training - query/set training flag given model or module layer
// trainflag - set training flag & return previous setting
// -------------------------------------------------------------------------------------------
KAPI training(K x) {
 KTRY
  bool b; Ktag *g;
  TORCH_CHECK((g=xtag(x)) || ((g=xtag(x,0)) && x->n==2 && xbool(x,1,b)),
              "training: unrecognized arg(s), expects module/model and optional flag");
  TORCH_CHECK(g->a==Class::module || g->a==Class::model, "training: not implemented for ",mapclass(g->a));
  return (x->n==2) ? g->module().train(b),(K)0 : kb(g->module().is_training());
 KCATCH("training");
}

static bool trainflag(Module& m,bool b) { bool a=m.is_training(); if(a != b) m.train(b); return a;}

// ------------------------------------------------------------------------------------------
// clipgrad - given vector of tensors, clip gradients by value/norm
// clipgroup - clip norm by optimizer group, return k list
// modelclip - clip using model training options (from within training loop -- no result)
// batchclip - clip using model from k session, return  double scalar/list
// kclip - handle input ptr to allocated tensor/vector/dict/module/model and args
// clipv - api function for clipping gradient value from k session
//  clip - api function for clipping gradient given model only or w'explicit norm args
// ------------------------------------------------------------------------------------------
static double clipgrad(bool a,F f,F p,const TensorVector& v) {
 if(a)
  return torch::nn::utils::clip_grad_norm_(v,f,p);
 else
  return torch::nn::utils::clip_grad_value_(v,f), nf;
}

static K clipgroup(const Optimizer& o,double f,double p) {
 const auto& g=o.param_groups();
 K r=ktn(KF,g.size()); J i=0;
 for(const auto& z:g) kF(r)[i++]=clipgrad(true,f,p,z.params());
 return r;
}

static void modelclip(Kmodel *m) {
 if(auto a=m->train.clipnorm()) {
  const auto& f=a.value();
  if(m->train.clipgroup()) {
   for(const auto& g:m->opt().param_groups())
    clipgrad(true, f[0], f[1], g.params());
  } else {
   clipgrad(true, f[0], f[1], m->kopt()->module().parameters());
  }
 } else if(auto a=m->train.clipvalue()) {
  clipgrad(false, a.value(), 0, m->kopt()->module().parameters());
 }
}

static K batchclip(Kopt* o,bool a,bool b,double f,double p) {
 return (a && b) ? clipgroup(o->opt(),f,p) : kf(clipgrad(a,f,p,o->module().parameters()));
}

static K batchclip(Kmodel *m) {
 KTRY
  if(m->train.clipnorm()) {
   const auto& f=*m->train.clipnorm();
   return batchclip(m->kopt(),true,m->train.clipgroup(),f[0],f[1]);
  } else if(m->train.clipvalue()) {
   return batchclip(m->kopt(),false,false,*m->train.clipvalue(),0);
  } else {
   return (K)0;
  }
 KCATCH("clip");
}

static K kclip(K x,bool a,const char *c) {
 KTRY
  Ktag *g=xtag(x,0); F f,p=2.0; bool b=false;
  TORCH_CHECK(g, c,": expects tensor(s), module, model or optimizer as 1st of 2",(a ? "-4" : "")," args");
  if(a) {
   TORCH_CHECK(x->n>1 && x->n<5, c,": expects 2-4 args, (",mapclass(g->a),"; max norm; norm exponent; group flag)");
  } else {
   TORCH_CHECK(x->n==2, c,": expects 2 args, (",mapclass(g->a),"; value)");
  }
  TORCH_CHECK(xnum(x,1,f), c,": 2nd arg, ",(a ? "max norm" : "max value"),", is ",kname(x,1),", expecting long/double");
  if(x->n==3) {
   TORCH_CHECK(xnum(x,2,p) || xbool(x,2,b), c,": 3rd arg, group flag or norm exponent expected, given ",kname(x,2));
  } else if(x->n==4) {
   TORCH_CHECK(xnum(x,2,p), c,": 3rd arg, norm exponent, is ",kname(x,2),", expecting long/double");
   TORCH_CHECK(xbool(x,3,b), c,": 4th arg, group flag, is ",kname(x,3),", expecting boolean");
  }
  switch(g->a) {
   case Class::tensor:    return kf(clipgrad(a,f,p,TensorVector{g->tensor()}));
   case Class::vector:    return kf(clipgrad(a,f,p,g->vector()));
   case Class::dict:      return kf(clipgrad(a,f,p,g->dict().values()));
   case Class::module:    return kf(clipgrad(a,f,p,g->module().parameters()));
   case Class::model:     return batchclip(((Kmodel*)g)->kopt(), a,b,f,p);
   case Class::optimizer: return batchclip((Kopt*)g, a,b,f,p);
   default: TORCH_ERROR(c,": not implemented for ",mapclass(g->a));
  }
 KCATCH(c);
}

KAPI clipv(K x) {return kclip(x, false, "clip gradient value");}

KAPI clip(K x) {
 if(auto *m=xmodel(x))
  return batchclip(m);
 else
  return kclip(x, true, "clip gradient norm");
}

// -------------------------------------------------------------------------------
// firstdevice - find 1st device of tensor(s) in input(s) or parameters of a model
// -------------------------------------------------------------------------------
static Device firstdevice(const Input& x,const Input& y) {
 auto d=firstdevice(x);
 return d ? *d : defaultdevice(firstdevice(y));
}

static Device firstdevice(Ktag *g) {
 if(g && (g->a==Class::model || g->a==Class::module)) {
  if(!g->kmodule()->d)
   g->kmodule()->d=defaultdevice(firstdevice(g->module().parameters()));
  return g->kmodule()->d.value();
 } else {
  return defaultdevice(c10::nullopt);
 }
}

// -------------------------------------------------------------------------
// tensorinput/vectorinput/dictinput - add tensor/vector/dictionary to input
// -------------------------------------------------------------------------
static void tensorinput(Input& x,const Tensor& y,const char *z) {
 if(c10::get_if<Empty>(&x)) {
  x=y;
 } else if(auto a=c10::get_if<Tensor>(&x)) {
  x=TensorVector({*a,y});
 } else if(auto a=c10::get_if<TensorVector>(&x)) {
  a->emplace_back(y);
 } else if(c10::get_if<TensorDict>(&x)) {
  TORCH_ERROR(z,": unable to merge tensor with previously specified dictionary");
 } else {
  TORCH_ERROR(z,": unrecognized input state");
 }
}

static void vectorinput(Input& x,const TensorVector& y,const char *z) {
 if(c10::get_if<Empty>(&x)) {
  x=y;
 } else if(auto a=c10::get_if<Tensor>(&x)) {
  TensorVector v({*a}); v.insert(v.end(),y.begin(),y.end()); x=v;
 } else if(auto a=c10::get_if<TensorVector>(&x)) {
  a->insert(a->end(),y.begin(),y.end());
 } else if(c10::get_if<TensorDict>(&x)) {
  TORCH_ERROR(z,": unable to merge dictionary with previously specified tensor(s)");
 } else {
  TORCH_ERROR(z,": unrecognized input state");
 }
}

static void dictinput(Input& x,const TensorDict& y,const char *z) {
 if(c10::get_if<Empty>(&x)) {
  x=y;
 } else if(c10::get_if<Tensor>(&x) || c10::get_if<TensorVector>(&x)) {
  TORCH_ERROR(z,": unable to merge dictionary with previously specified tensor(s)");
 } else if(auto a=c10::get_if<TensorDict>(&x)) {
  a->update(y);
 } else {
  TORCH_ERROR(z,": unrecognized input state");
 }
}

// --------------------------------------------------------------------
// inputpair: check for vector/dict and index(es)/key(s)
// --------------------------------------------------------------------
static bool inputpair(K x,Input& in,const char* c) {
 bool b=false;
 if(!x->t && x->n==2) {
  K y=kK(x)[1]; auto *v=xvec(x,0); auto *d=v ? nullptr : xtensordict(x,0);
  if(v) {
   IntArrayRef n; int64_t m=v->size();
   TORCH_CHECK(xsize(y,n), c,": vector paired with ",kname(y)," expecting long index or indices");
   for(const auto i:n) {
    TORCH_CHECK(-1<i && i<m, c,": vector[",i,"] invalid for ",m,"-element vector");
    tensorinput(in,v->at(i),c);
   }
   b=true;
  } else if(d) {
   SymArrayRef s; Tensor *t;
   TORCH_CHECK(xsyms(y,s), c,": dictionary paired with ",kname(y)," expecting symbol key(s)");
   for(const auto& k:s) {
    TORCH_CHECK(t=d->find(k),"dictionary key: `",k," not found");
    tensorinput(in,*t,c);
   }
   b=true;
  }
 }
 return b;
}

// ----------------------------------------------------------------------------
// modelarg - handle k input(s) for forward calculations
// ----------------------------------------------------------------------------
static void modelarg(K x,const char *c,Ktag *g,Input& in) {
 if(auto *a=xten(x)) {
  tensorinput(in,*a,c);
 } else if(auto *a=xvec(x)) {
  vectorinput(in,*a,c);
 } else if(auto *a=xtensordict(x)) {
  dictinput(in,*a,c);
 } else if(xarray(x,7)) {
  tensorinput(in,kput(x).to(firstdevice(g)),c);
 } else if(inputpair(x,in,c)) {
 } else {
  TORCH_CHECK(!x->t, c,": unrecognized arg, ",kname(x));
  for(J i=0; i<x->n;++i)
    modelarg(kK(x)[i],c,g,in);
 }
}

Input modelarg(K x,J i,const char *c) {
 Ktag *g=xtag(x,0); Input in=Empty();
 for(;i<x->n; ++i) 
  modelarg(kK(x)[i],c,g,in);
 return in;
}

// ---------------------------------------------------------------------------
// modelargs - handle k arg parsing for model w'both inputs & targets
// ---------------------------------------------------------------------------
std::tuple<Input,Input> modelargs(K z,const char* c) {
 TORCH_CHECK(z->n<4, c,": up to 3 args expected, (",kname(z,0),";inputs;targets) but ",z->n," given");
 auto g=xtag(z,0); Input x=Empty(); modelarg(kK(z)[1],c,g,x);
 if(z->n==2) {
  return std::make_tuple(x, Empty());
 } else {
  z=kK(z)[2]; Input y=Empty(); modelarg(z,c,g,y);
  return std::make_tuple(x,y);
 }
}

// -----------------------------------------------------------------------------------------
//  submodule - returns a child module referenced by name with additional attributes
//            - used in forward call, e.g. forward(m;`k; ..) to avoid managing module ptr
// -----------------------------------------------------------------------------------------
static Kmodule submodule(const Module& m,S s) {
 if(strchr((C*)s,'.')) {
  const Moduleptr& p=m.named_modules()[s];
  return Kmodule(Class::module,mcast(p),p);
 } else {
  const Moduleptr p=m.named_children()[s];   // named_children needs extra reference
  return Kmodule(Class::module,mcast(p),p);
 }
}

// -----------------------------------------------------------------------------------------
//  qforward - utility fns for managing forward calc in training & evaluation mode
//   forward - k api fn for forward calcs given module/model & inputs, returns tensor(s)
//  eforward - k api fn for forward calcs with no grad mode & training mode off
//  evaluate - same as eforward, but returns k array(s) instead of tensor(s)
//  kforward - used in callback fn: calls forward/eforward if module training true/false
// -----------------------------------------------------------------------------------------
static K qforward(Kmodule *m,bool b,bool g,bool k,const Input& x) {
// b-true if training, g-true if gradients, k-true if k value returned
 torch::AutoGradMode grad(g);
 bool a=trainflag(m->module(),b);
 K r=k ? kget(mforward(m,x)) : kout(mforward(m,x));
 m->module().train(a);
 return r;
}

static K qforward(K x,bool a,bool b,bool g,bool k,const char *c) { 
// a-true for k callback, b-true if training, g-true if gradients, k-true if k value returned
 KTRY
  Kmodel *m=xmodel(x); S s=nullptr;
  if(m) {
   TORCH_CHECK(!a, "kforward: cannot be called without input(s)");
   return qforward(m->kmodule(), b, g, k, (b ? m->data : m->testdata).x);
  } else {
   m=xmodel(x,0); auto *q=m ? m->kmodule() : xmodule(x,0); J i=xsym(x,1,s) ? 2 : 1;
   TORCH_CHECK(q && x->n>i, c,": requires a module or model and at least one input");
   // if forward call within a module callback from k, training & gradients set by parent environment
   if(a) b=q->module().is_training(), g=torch::GradMode::is_enabled(); 
   if(s) {
    auto m=submodule(q->module(),s);
    return qforward(&m, b, g, k, modelarg(x,i,c));
   } else {
    return qforward(q, b, g, k, modelarg(x,i,c));
   }
  }
 KCATCH(c);
}

KAPI  forward(K x) { return qforward(x, false, true,  true,  false, "forward"); }   //training, gradients, return tensor(s)
KAPI nforward(K x) { return qforward(x, false, true,  false, false, "forward"); }   //training w'out gradients, return tensor(s)
KAPI eforward(K x) { return qforward(x, false, false, false, false, "eforward"); }  //evaluate, no gradients, return tensor(s)
KAPI evaluate(K x) { return qforward(x, false, false, false, true,  "evaluate"); }  //evaluate, no gradients, return k arrays
KAPI kforward(K x) { return qforward(x, true,  false, false, false, "kforward"); }  //k callback: uses parent training/eval & gradients

// -----------------------------------------------------------------------------------------
// mbackward - given model, inputs & targets, calculate loss from model outputs and targets
// tbackward - backprop given tensor, optional tensor & sym for retain/create gradient graph
//  backward - k api function for backward calc on model/tensor, return loss/null
// -----------------------------------------------------------------------------------------
K mbackward(Kmodel *m,const Input& x,const Input& y) {
 auto t=losscalc(m, mforward(m->kmodule(),x), y);
 t.backward();
 return kget(t);
}

static K tbackward(K x,Tensor& t) {
 bool a=false,b=false; J n=x->n-xbacksym(x,x->n-1,a,b);
 if(n==1) {
  t.backward({},a,b);
 } else if(n==2) {
  Tensor g;
  if(!xten(x,1,g)) g=kput(x,1).to(t.device());
  if(!g.dim() && t.dim()) g.resize_as_(t).fill_(g[0]);
  t.backward(g,a,b);
 } else {
  TORCH_ERROR("backward: unexpected arg(s), expecting (t;s), (t;g) or (t;g;s) with t-tensor, g-tensor, s-sym, e.g. `retain");
 }
 return (K)0;
}

KAPI backward(K a) {
 KTRY
  if(auto m=xmodel(a)) {
   return mbackward(m,m->data.x,m->data.y);
  } else if(auto m=xmodel(a,0)) {
   Input x,y; std::tie(x,y)=modelargs(a,"backward");
   return mbackward(m,x,y);
  } else if(auto t=xten(a)) {
   return t->backward(), (K)0;
  } else if(auto t=xten(a,0)) {
   return tbackward(a,*t);
  } else {
   TORCH_ERROR("backward: expects tensor or model as 1st arg, e.g. backward(tensor) or backward(model;inputs;targets)");
  }
 KCATCH("backward");
}

// ----------------------------------------------------------------------------
// fullsize -- undo any batching on input(s) & target(s)
// ----------------------------------------------------------------------------
static int64_t fullinput(const Input& x,int64_t d,int64_t n) {
 return c10::visit(
  c10::overloaded(
   [&](const auto& x)  {return fullsize(x,d,n);},
   [&](const Empty& x) {return int64_t(0);}
  ),x);
}

static int64_t fullsize(Data& d) {
 d.batch(-1); auto n=d.size();
 fullinput(d.x, 0, n);
 fullinput(d.y, 0, n);
 return n;
}

// ----------------------------------------------------------------------------
// reindex - given permutation index, reorder input(s) & target(s)
// reshuffle - create new random permutation, reorder tensor(s)
// ----------------------------------------------------------------------------
static void reindex(Tensor& t,const Tensor& i,int64_t d=0) {
 if(t.defined())
  t=t.index_select(d,i.to(t.device()));
}

static void reindex(TensorVector& v,const Tensor& i,int64_t d=0) {
 for(auto& t:v)
  reindex(t,i,d);
}

static void reindex(TensorDict& x,const Tensor& i,int64_t d=0) {
 for(auto& y:x.items())
  if(y.value().defined())
   x[y.key()]=y.value().index_select(d,i.to(y.value().device()));
}

static void reindex(Input& x,const Tensor& i,int64_t d=0) {
 c10::visit(
  c10::overloaded(
   [&](Tensor& x)       {reindex(x,i,d);},
   [&](TensorVector& x) {reindex(x,i,d);},
   [&](TensorDict& x)   {reindex(x,i,d);},
   [&](Empty& x)        {}
  ),x);
}

static Tensor perm(int64_t n,Generator& g) {
 return torch::randperm(n, g, torch::dtype(torch::kLong).device(g.device()));
}

static Tensor perm(int64_t n,const Device& d) {
 return torch::randperm(n, torch::dtype(torch::kLong).device(d));
}

static void reshuffle(Tensor& t, int64_t d=0) {
 if(t.defined())
  if(auto n=fullsize(t,d))
   reindex(t, perm(n, t.device()), d);
}

template<typename T>static void reshuffle(T& t, int64_t d=0) {
 if(t.size()) {
  if(auto n=fullsize(t,d))
   reindex(t, perm(n, defaultdevice(firstdevice(t))), d);
 }
}

static bool reshuffle(const  TestOptions& o,Data& d) {return false;}

static bool reshuffle(const TrainOptions& o,Data& d) {
 if(o.shuffle()) {
  auto n=fullsize(d);
  if(n) {
   Tensor i; auto c=firstdevice(d.x,d.y);
   if(o.tasks()<2) {
    i=perm(d.size(), c);
   } else {
    if(!d.g.defined()) {  // set generator so different tasks can use same permutation
     d.g=torch::globalContext().defaultGenerator(o.shufflecuda() ? c : Device(DeviceType::CPU)).clone();
     d.g.set_current_seed(o.shuffleseed());
    }
    i=perm(d.size(), d.g);
   }
   reindex(d.x,i); reindex(d.y,i);
   d.p = d.p.defined() ? d.p.index_select(0,i.to(d.p.device())) : i;
  }
  return true;
 } else {
  return false;
 }
}

// ----------------------------------------------------------------------------
// newmetrics - [re]init a vector of vectors to accumulate tensors per batch
// batchinit - reset any previous batching, shuffle if required, reset metrics
// ----------------------------------------------------------------------------
static void newmetrics(size_t m,int64_t i,int64_t j,Data& d) {
 // number of metrics, i-task, j-number of tasks
 auto n=d.batches();        // number of batches
 d.m = MetricData(m);       // vector of tensors for each metric
 auto v = n/j + (i < n%j);  // number of vector elements for each metric
 for(size_t i=0; i<m; ++i)
  if(n>-1)
   d.m[i]=TensorVector(v);  // reserve space for tensors from each batch
}

template<typename O> static void batchinit(const O& o,Data& d) {
 if(!reshuffle(o,d)) fullsize(d);
 newmetrics(o.metrics().size(), o.task(), o.tasks(), d);
}

KAPI shuffle(K x) {
 KTRY
  int64_t d=0;
  TORCH_CHECK(!x->t, "shuffle: not implemented for ",kname(x));
  TORCH_CHECK(0<x->n && x->n<3, "shuffle: expecting 1-2 args, tensor/vector/dictionary/model & optional dimension, but given ",x->n," args");
  TORCH_CHECK(x->n==1 || xint64(x,1,d), "shuffle: 2nd arg is dimension, but given ",kname(x,1));
  bool b=x->n==1;
  if(auto a=b ? xten(x) : xten(x,0)) {                       reshuffle(*a,d);
  } else if (auto a=b ? xvec(x) : xvec(x,0)) {               reshuffle(*a,d);
  } else if (auto a=b ? xtensordict(x) : xtensordict(x,0)) { reshuffle(*a,d);
  } else if (auto a=b ? xmodel(x) : xmodel(x,0)) {
   TORCH_CHECK(d==0, "shuffle: model data is only batched on dimension 0, but given dimension ",d);
   reshuffle(a->train,a->data);
  } else {
   TORCH_ERROR("shuffled: expecting tensor,vector,dictionary or model, given ",kname(x));
  }
  return (K)0;
 KCATCH("shuffle");
}

KAPI unshuffle(K x) {
 KTRY
  Kmodel *m=xmodel(x);
  TORCH_CHECK(m, "unshuffle: requires model argument, given ",kname(x));
  auto &d=m->data;
  fullsize(d);
  if(d.p.defined()) {
   auto i=d.p.argsort(); reindex(d.x,i); reindex(d.y,i);
   d.p=Tensor();
  }
  return (K)0;
 KCATCH("unshuffle");
}

// -----------------------------------------------------------------------------
// batches - calculate number of batches given overall data size & batch size
// datainit - [re]assign input(s) & target(s) for training/testing
// nextbatch - true if more data, setting batch size for model inputs & targets
// batch/testbatch - k api functions to process next batch
// -----------------------------------------------------------------------------
static void batches(Data& d,bool b,int64_t n,int64_t w) {
 if(w>n) w=n;
 d.size(n);                 // size of tensors along batch dim (typically 1st dim)
 d.batchsize(w);            // batch size from specified train/test options
 d.batches(batches(w,n,b)); // save number of batches (w'option to omit final partial batch)
 d.batch(-1);               // current batch set to indicate none selected yet
}

static void batches(Data& d,bool b,int64_t w) {
 if(d.size() > -1)
  batches(d,b,d.size(),w);
}


template<typename O> static int64_t datainit(O& o,Data& d,const Input& x,const Input &y) {
 // both input(s) & target(s) specified for model train/test
 batches(d,o.droplast(),checksize(x,y),o.batchsize());
 d.x = std::move(x);   // model input(s)
 d.y = std::move(y);   // target(s)
 d.p = Tensor();       // permutation index (used if shuffling training data)
 d.g = Generator();    // generator used for permutations
 return d.batches();   // return number of batches
}

template<typename O> static int64_t datainit(O& o,Data& d,bool b,const Input& z) {
 // b flag true if defining input(s) else target(s) for model train/test
 batches(d,o.droplast(),checksize(b ? z : d.x, b ? d.y : z),o.batchsize());
 if(b) d.x=std::move(z); else d.y=std::move(z);
 d.p = Tensor();       // permutation index (used if shuffling training data)
 return d.batches();   // return number of batches
}

template<typename O> static bool nextbatch(const O& o,Data &d) {
 auto i=d.batch(); bool b=i<0; i=b ? o.task() : i + o.tasks();
 if(i < d.batches()) {
  if(b) batchinit(o,d);
  batch(d.x, i, d.batchsize(), 0, d.size()); // select i'th batch of input(s)
  batch(d.y, i, d.batchsize(), 0, d.size()); // select i'th batch of target(s)
  d.batch(i);                                // [re]set current batch index
  return true;
 } else {
  fullsize(d);
  return false;
 }
}

static K modelbatch(K x,bool b,const char *c) {
 KTRY
  Kmodel *m=xmodel(x);
  TORCH_CHECK(m, c,": expecting model as only argument, given ",kname(x));
  return kb(b ? nextbatch(m->train,m->data) : nextbatch(m->test,m->testdata));
 KCATCH(c);
}

KAPI trainbatch(K x) {
 KTRY
  TORCH_CHECK(!x->t && x->n, "batch: not implemented for ",kname(x));
  if(x->n==1) {
   return modelbatch(x,true,"batch");
  } else {
   IntArrayRef n;
   TORCH_CHECK(x->n>1 && x->n<4, "batch: expecting 2-3 args, given ",x->n);
   TORCH_CHECK(xsize(x,1,n), "batch: 2nd arg is batch size or batch size & dimension, but given ",kname(x,1));
   TORCH_CHECK(n.size() && n.size()<3, "batch: 2nd arg is batch size or batch size & dimension, but given ",n.size()," elements");
   auto w=n[0],d=n.size()==1 ? 0 : n[1];
   TORCH_CHECK(d>-1, "batch: dimension cannot be negative");
   if(x->n==2) {
    return kb(nextbatch(kK(x)[0],w,d));
   } else {
    int64_t i;
    TORCH_CHECK(xint64(x,2,i), "batch: 3rd arg is batch index, but given ",kname(x,2));
    return batchindex(kK(x)[0],w,d,i), (K)0;
   }
  }
 KCATCH("batch");
}

KAPI testbatch(K x) {return modelbatch(x, false, "testbatch");}

// ---------------------------------------------------------------------------------
// batchinit - reset batches on train/test data defined for the model
// ---------------------------------------------------------------------------------
static K batchinit(K x,bool b,const char *c) {
 KTRY
  Kmodel *m=xmodel(x);
  TORCH_CHECK(m, c,": expected model as argument, given ",kname(x));
  if(b)
   batchinit(m->train,m->data);
  else
   batchinit(m->test,m->testdata);
  return kj(b ? m->data.batches() : m->testdata.batches());
 KCATCH(c);
}

KAPI Batchinit(K x) {return batchinit(x, true,  "batchinit");}
KAPI  testinit(K x) {return batchinit(x, false, "testinit");}

// ---------------------------------------------------------------------------
// output - retrieve output tensor from model output (vector,tuple, etc.)
// hidden - retrieve hidden state, hidden cell state from model output
// matches - return count where prediction matches target and overall count
// ---------------------------------------------------------------------------
static Tensor output(Kmodel *m,const Output& x) {
 return c10::visit(
  c10::overloaded(
   [&](const Tensor& x)       -> Tensor {return x;},
   [&](const TensorVector& x) -> Tensor {TORCH_CHECK(x.size(), "unable to retrieve model output from empty vector of tensors"); return x[0];},
   [&](const Tuple& x)        -> Tensor {return std::get<0>(x);},
   [&](const Nested& x)       -> Tensor {return std::get<0>(x);},
   [&](const auto& x)         -> Tensor {TORCH_ERROR("unable to retrieve model output from ",outputname(x)," output");}
  ),x);
}

static Tensor hidden(Metric m,const Output& x) {
 if(auto a=c10::get_if<TensorVector>(&x)) {
  if(a->size()>1 && m==Metric::hidden)          return (*a)[1];
  else if(a->size()>2 && m==Metric::hiddencell) return (*a)[2];
  else TORCH_ERROR("metric: unable to retrieve ",metric(m)," from vector of tensors with ",a->size()," element(s)");
 } else if(auto a=c10::get_if<Tuple>(&x)) {
  return std::get<1>(*a);
 } else if(auto a=c10::get_if<Nested>(&x)) {
  return m==Metric::hidden ? std::get<0>(std::get<1>(*a)) : std::get<1>(std::get<1>(*a));
 } else {
  TORCH_ERROR("metric: unable to retrieve ",metric(m)," from ",outputname(x)," output");
 }
}

static Tensor matches(Kmodel *m,const Tensor& yhat,const Input &y) {
 if(auto a=c10::get_if<Tensor>(&y)) {
  return torch::stack({a->eq(yhat).sum(),
                       torch::tensor(a->numel(),torch::device(a->device()))}).view({1,2});
 } else {
  TORCH_ERROR("unable to calculate matches from ",inputname(y)," of target(s)");
 }
}

// -----------------------------------------------------------------------------------
// lossflag - return true if loss calculation required
// metrics - given input(s), target(s) & model output for the batch, calculate metrics
// -----------------------------------------------------------------------------------
static bool lossflag(const Metrics& m) {
 for(auto k:m)
  if(k==Metric::loss || k==Metric::batchloss) return true;
 return false;
}

static void metrics(Kmodel *m,const Metrics& k,int64_t n,Data& d) {
  auto j=d.batch() / n;          // assign metric[i][j] with n tasks
  if(k.size()) {                 // if metrics defined
   size_t i=0; Tensor out,pred;  // tensors for output, prediction
   for(auto c:k) {
    switch(c) {
     case Metric::output:
      if(!out.defined()) out=output(m,d.z);
      // if network is identity function, out -> d.z -> d.x, i.e. a part of input
      // without clone, when input(s) resized, stored metric is also resized
      d.m[i][j]=(out.defined() && out.use_count()>2) ? out.clone() : out;
      break;
     case Metric::loss:
     case Metric::batchloss:
      d.m[i][j]=d.l.detach();
      break;
     case Metric::predict:
      if(!out.defined()) out=output(m,d.z);
      if(!pred.defined()) pred=out.argmax(-1);
      d.m[i][j]=pred.detach();
      break;
     case Metric::accuracy:
     case Metric::matches:
      if(!out.defined()) out=output(m,d.z).detach();
      if(!pred.defined()) pred=out.argmax(-1).detach();
      d.m[i][j]=matches(m,pred,d.y).detach();
      break;
     case Metric::hidden:
     case Metric::hiddencell:
      d.m[i][j]=hidden(c,d.z).detach();
      break;
     default:
      TORCH_ERROR("metric: ",metric(c)," not recognized or not implemented");
    }
    i++;
   }
  }
}

// ---------------------------------------------------------------------------
// hiddenstate - combine previous hidden state w'input for model forward calc
// ---------------------------------------------------------------------------
TensorVector hiddenstate (Data& d) {
 return c10::visit(
  c10::overloaded (
   [](const Tensor& x,const TensorVector& y) {
    switch(y.size()) {
     case 2:   return TensorVector({x,y[1]});
     case 3:   return TensorVector({x,y[1],y[2]}); 
     default: TORCH_ERROR("unable to retrieve hidden state from ",y.size(), "-element vector of output(s)");
    }
   },
   [](const Tensor& x,const Tuple& y) {return TensorVector({x,std::get<1>(y)});},
   [](const Tensor& x,const Nested& y) {
     return TensorVector({x, std::get<0>(std::get<1>(y)), std::get<1>(std::get<1>(y))});},
   [](auto x,auto y) {
     TORCH_ERROR("unable to retrive hidden state from ",inputname(x)," input & ",outputname(y)," output");
      return TensorVector();}
  ), d.x, d.z);
}

// ---------------------------------------------------------------------------
// batchcalc - run forward calcs, get loss if required, calculate metrics
// trainstep - reset grad,forward,loss,backward & step using closure
// trainloop/testloop - run model in train/evaluate mode, accumulate metrics
// backstep - calculate outputs,loss,gradients and perform optimizer step
// ---------------------------------------------------------------------------
template<typename O> static void batchcalc(Kmodel *m,bool b,const O& o,Data& d) {
 d.z=mforward(m->kmodule(), o.hidden() && d.batch() > o.task() ? hiddenstate(d) : d.x);
 if(b) d.l=losscalc(m, d.z, d.y);
 metrics(m, o.metrics(), o.tasks(), d);
}

static void trainstep(Kmodel *m,const Input& x,const Input& y) {
 auto f=[&]() {
  auto& d=m->data;
  resetgrad(m->opt(),true);
  d.z=mforward(m->kmodule(), x);
  auto l=losscalc(m, d.z, y);
  l.backward();
  if(m->train.sync()) sync(l.device());
  modelclip(m);
  return l;
 };
 m->data.l=m->opt().step(f);
}

static void trainstep(Kmodel *m) {
 auto& d=m->data;
 trainstep(m, m->train.hidden() && d.batch() > m->train.task() ? hiddenstate(d) : d.x, d.y);
}

static void trainloop(Kmodel *m) {
 bool a=trainflag(m->module(),true); auto& d=m->data; const auto& o=m->train;
 while(nextbatch(o,d)) {
  trainstep(m);
  metrics(m, o.metrics(), o.tasks(), m->data);
 }
 m->module().train(a); fullsize(d);
}

static void testloop(Kmodel *m) {
 torch::NoGradGuard g;
 bool a=trainflag(m->module(),false); auto& d=m->testdata; const auto& o=m->test;
 bool b=lossflag(o.metrics());
 while(nextbatch(o,d))
  batchcalc(m,b,o,d);
 m->module().train(a); fullsize(d);
}

KAPI backstep(K a) {
 KTRY
  Tensor t; auto m=xmodel(a);
  if(m) {
   trainstep(m);
  } else {
   m=xmodel(a,0);
   TORCH_CHECK(m, "backstep: model expected as 1st argument, given ",kname(a,0));
   TORCH_CHECK(a->n==3, "backstep expecting (model;inputs;targets), but ",a->n," args given");
   Input x,y; std::tie(x,y)=modelargs(a,"backstep");
   trainstep(m,x,y);
  }
  return kget(m->data.l);
 KCATCH("backstep");
}

// ---------------------------------------------------------------------------
//  getmetrics - catenate/stack batch metrics into vector of tensors
//               return as k values or pytorch tensor/vector/dictionary
// ---------------------------------------------------------------------------
TensorVector getmetrics(Data& d) {
 TensorVector r;
 for(const auto& v:d.m)
  if(v.size())
   r.emplace_back(v[0].dim() ? torch::cat(v) : torch::stack(v));
  else
   r.emplace_back(Tensor());
 d.m = MetricData();
 return r;
}
 
template<typename O> static K getmetrics(const O& o,Data& d) {
 size_t i=0; TensorVector v=getmetrics(d);
 if(!v.size()) {                             // no metrics recorded(no data supplied?)
  for(size_t i=0; i<o.metrics().size(); ++i) // for each metric in settings
   v.emplace_back(Tensor());                 // use undefined tensor
 } else {
  for(auto m:o.metrics()) {
   if(m==Metric::accuracy || m==Metric::matches) {
    auto s=v[i].sum(0);
    v[i] = m==Metric::accuracy ? 100.0*s[0].div(s[1]) : s;
   } else if(m==Metric::loss) {
    v[i] = v[i].sum().to(torch::kDouble) / d.batches();
   }
   i++;
  }
 }
 if(o.dictionary()) {
  if(o.tensor()) {
   TensorDict d; size_t i=0;
   for(auto m:o.metrics())
    d.insert(metric(m), v[i++]);
   return kdict(d);
  } else {
   K x=KDICT; size_t i=0;
   for(auto m:o.metrics())
    dictadd(x, metric(m), kget(v[i++]));
   return x;
  }
 } else {
  if(o.tensor())
   return v.size()==1 ? kten(v[0]) : kvec(v);
  else
   return v.size()==1 ? kget(v[0]) : kget(v);
 }
}

// -----------------------------------------------------------------
// modelopt - translate between model option symbol and enumeration
// -----------------------------------------------------------------
static Setting modelopt(S s,bool t) {
 for(const auto& a:env().train)
  if(std::get<0>(a)==s && (t ? std::get<2>(a) : std::get<3>(a)))
   return std::get<1>(a);
 TORCH_ERROR("unrecognized ", (t ? "train" : "test")," setting: ",s);
}

static S modelopt(Setting s) {
 for(const auto& a:env().train)
  if(std::get<1>(a)==s) return std::get<0>(a);
 TORCH_ERROR("unrecognized training/evaluation setting: ",(I)s);
}

// ----------------------------------------------------------------------
// getoption - retrieve individual option, list or full set as dictionary
// ----------------------------------------------------------------------
static K getoption(Kmodel *m,bool t,Setting s) {
 switch(s) {
  case Setting::batchsize:   return kj(t ? m->train.batchsize()       : m->test.batchsize());
  case Setting::droplast:    return kb(t ? m->train.droplast()        : m->test.droplast());
  case Setting::hidden:      return kb(t ? m->train.hidden()          : m->test.hidden());
  case Setting::tensor:      return kb(t ? m->train.tensor()          : m->test.tensor());
  case Setting::dictionary:  return kb(t ? m->train.dictionary()      : m->test.dictionary());
  case Setting::shuffle:     return t    ? kb(m->train.shuffle())     : nullptr;
  case Setting::shufflecuda: return t    ? kb(m->train.shufflecuda()) : nullptr;
  case Setting::shuffleseed: return t    ? kj(m->train.shuffleseed()) : nullptr;
  case Setting::sync:        return t    ? kb(m->train.sync())        : nullptr;
  case Setting::task:        return kj(t ? m->train.task()            : m->test.task());
  case Setting::tasks:       return kj(t ? m->train.tasks()           : m->test.tasks());
  case Setting::clipgroup:   return t    ? kb(m->train.clipgroup())   : nullptr;
  case Setting::clipvalue:   return t && m->train.clipvalue() ? kf(*m->train.clipvalue()) : nullptr;
  case Setting::clipnorm:
   if(t && m->train.clipnorm()) {
    K y=ktn(KF,2); const auto& z=*m->train.clipnorm(); kF(y)[0]=z[0]; kF(y)[1]=z[1];
    return y;
   } else {
    return nullptr;
   }
  case Setting::metrics: {
   const auto& v=t ? m->train.metrics() : m->test.metrics();
   K y=ktn(KS,v.size()); J j=0; for(auto i:v) kS(y)[j++]=metric(i);
   return y;
  }
  default:
   TORCH_ERROR((t ? "train" : "test"),": unrecognized setting");
 }
}

static K getoption(Kmodel *m,bool t,SymArrayRef a) {
  K y=KDICT;
  try {
   for(auto k:a) {
    K z=getoption(m,t,modelopt(k,t));
    dictadd(y, k, z ? z : knull());
   }
   return resolvedict(y);
  } catch(...) {
   if(y) r0(y);
   throw;
  }
}

static K getoption(Kmodel *m,bool t,bool a) {
 K x=KDICT; K y;
 OPTION(x, batchsize, getoption(m,t,Setting::batchsize));
 OPTION(x, task,      getoption(m,t,Setting::task));
 OPTION(x, tasks,     getoption(m,t,Setting::tasks));
 OPTION(x, droplast,  getoption(m,t,Setting::droplast));
 OPTION(x, hidden,    getoption(m,t,Setting::hidden));
 if(t) {
  OPTION(x, shuffle,     getoption(m,t,Setting::shuffle));
  OPTION(x, shufflecuda, getoption(m,t,Setting::shufflecuda));
  OPTION(x, shuffleseed, getoption(m,t,Setting::shuffleseed));
  OPTION(x, sync,        getoption(m,t,Setting::sync));
  OPTION(x, clipgroup,   getoption(m,t,Setting::clipgroup));
  y=getoption(m,t,Setting::clipnorm);  if(a || y) OPTION(x, clipnorm,  y ? y : knull());
  y=getoption(m,t,Setting::clipvalue); if(a || y) OPTION(x, clipvalue, y ? y : knull());
 }
 OPTION(x, tensor,     getoption(m,t,Setting::tensor));
 OPTION(x, dictionary, getoption(m,t,Setting::dictionary));
 OPTION(x, metrics,    getoption(m,t,Setting::metrics));
 return x; 
}

// -----------------------------------------------------------------
// clipnorm - handle 1-2 long(s)/double(s) for clipping by norm
// checkopt - check for incompatible train/test options
// setoption - parse k arg(s) to set or reset model options
// -----------------------------------------------------------------
static void clipnorm(Kmodel *m,S a,K x) {
 F f,p=2.0;
 if(xempty(x)) {
  m->train.clipnorm(c10::nullopt);
 } else {
  if(xnum(x,f) || (xnum(x,0,f) && xnum(x,1,p) && x->n==2)) {
  } else if (x->t==KF || x->t==KJ) {
   TORCH_CHECK(x->n==2, a,": expecting 1-2 numbers, given ",x->n);
   if(x->t==KF)
    f=kF(x)[0], p=kF(x)[1];
   else
    f=kJ(x)[0], p=kJ(x)[1];
  } else {
   TORCH_ERROR(a,": expecting 1-2 numbers, long integers or doubles, given ",kname(x));
  }
  m->train.clipnorm({{f,p}}).clipvalue(c10::nullopt);
 }
}

template<typename O> static void checkopt(const char *c,const O& o) {
 TORCH_CHECK(o.task() < o.tasks(), c,": task ",o.task()," implies at least ",o.task()+1," tasks, but only ",o.tasks()," defined");
 TORCH_CHECK(o.tasks()==1 || o.batchsize(), c,": zero batchsize incompatible with ",o.tasks()," tasks");
}

static void setoption(Kmodel *m,bool t,Setting s,S a,K x) {
 bool b; J n; F f;
 switch(s) {
  case Setting::batchsize:
   TORCH_CHECK(xlong(x,n), a,": expects one long integer, given ",kname(x));
   TORCH_CHECK(n>=0,       a,": cannot be less than zero, given ",n);
   if(t) m->train.batchsize(n), batches(m->data,m->train.droplast(),n);
   else  m->test.batchsize(n), batches(m->testdata,m->test.droplast(),n);
   break;
  case Setting::task:
   TORCH_CHECK(xlong(x,n), a,": expects one long integer, given ",kname(x));
   TORCH_CHECK(n>=0,       a,": cannot be less than zero, given ",n);
   if(t) m->train.task(n); else m->test.task(n);
   break;
  case Setting::tasks:
   TORCH_CHECK(xlong(x,n), a,": expects one long integer, given ",kname(x));
   TORCH_CHECK(n>0,        a,": cannot be less than one, given ",n);
   if(t) m->train.tasks(n); else m->test.tasks(n);
   break;
  case Setting::clipnorm:
   TORCH_CHECK(t, a,": not set for evaluation");
   clipnorm(m,a,x);
   break;
  case Setting::clipvalue:
   TORCH_CHECK(t, a,": not set for evaluation");
   if(xempty(x)) {
    m->train.clipvalue(c10::nullopt);
   } else {
    TORCH_CHECK(xnum(x,f), a,": expecting number for maximum gradient, given ",kname(x));
    m->train.clipvalue(f).clipnorm(c10::nullopt);
   }
   break;
  case Setting::clipgroup:
   TORCH_CHECK(t, a,": not set for evaluation");
   TORCH_CHECK(xbool(x,b), a,": expects one boolean, given ",kname(x));
   m->train.clipgroup(b);
   break;
  case Setting::dictionary:
   TORCH_CHECK(xbool(x,b), a,": expects one boolean, given ",kname(x));
   if(t) m->train.dictionary(b); else m->test.dictionary(b);
   break;
  case Setting::droplast:
   TORCH_CHECK(xbool(x,b), a,": expects one boolean, given ",kname(x));
   if(t) m->train.droplast(b), batches(m->data,b,m->train.batchsize());
   else  m->test.droplast(b), batches(m->testdata,b,m->test.batchsize());
   break;
  case Setting::hidden:
   TORCH_CHECK(xbool(x,b), a,": expects one boolean, given ",kname(x));
   if(t) m->train.hidden(b); else m->test.hidden(b);
   break;
  case Setting::metrics: {
   SymArrayRef sm; Metrics v;
   TORCH_CHECK(xempty(x)||xsyms(x,sm), a,": expects symbol(s), given ",kname(x));
   for(auto i:sm) v.push_back(metric(i));
   if(t) m->train.metrics(v); else m->test.metrics(v);
   break;
  }
  case Setting::shuffle:
   TORCH_CHECK(t, a,": not set for evaluation");
   TORCH_CHECK(xbool(x,b), a,": expects one boolean, given ",kname(x));
   m->train.shuffle(b);
   break;
  case Setting::shufflecuda:
   TORCH_CHECK(t, a,": not set for evaluation");
   TORCH_CHECK(xbool(x,b), a,": expects one boolean, given ",kname(x));
   m->train.shufflecuda(b);
   break;
  case Setting::shuffleseed:
   TORCH_CHECK(t, a,": not set for evaluation");
   TORCH_CHECK(xlong(x,n), a,": expects one long integer, given ",kname(x));
   m->train.shuffleseed(n);
   break;
  case Setting::sync:
   TORCH_CHECK(t, a,": not set for evaluation");
   TORCH_CHECK(xbool(x,b), a,": expects one boolean, given ",kname(x));
   m->train.sync(b);
   break;
  case Setting::tensor:
   TORCH_CHECK(xbool(x,b), a,": expects one boolean, given ",kname(x));
   if(t) m->train.tensor(b); else m->test.tensor(b);
   break;
  default: TORCH_ERROR("unrecognized setting: ",x);
 }
}

static void setoptions(Kmodel* m,bool t,const char *c,SymArrayRef a,K x) {
 J i=0; K y,z; auto tr=m->train; auto te=m->test;
 try {
  for(auto k:a) {
   y=nullptr,z=nullptr; auto s=modelopt(k,t);
   switch(x->t) {
    case 0:  y=kK(x)[i]; break;
    case KB: z=kb(kG(x)[i]); break;
    case KJ: z=kj(kJ(x)[i]); break;
    case KF: z=kf(kF(x)[i]); break;
    case KS: z=ks(kS(x)[i]); break;
    default: TORCH_ERROR(c,": unable to set options given ",kname(x)); break;
   }
   setoption(m,t,s,k,y ? y : z); i++; if(z) r0(z),z=nullptr;
  }
  if(t) checkopt("train", m->train); else checkopt("test", m->test);
 } catch(...) {
   if(t) m->train=tr; else m->test=te;  // restore previous options on error
   if(z) r0(z);                         // decrement ref if K value created here
   throw;
 }
}

// ----------------------------------------------------------------------------
// modelsym - get/set one training/evaluation option given sym & optional value
// modelopts - get/set options given multiple symbols or symbols & values
// modeldict - set options given dictionary of keys!values
// ---------------------------------------------------------------------------
static K modelsym(Kmodel *m,K x,bool t,S a) {
 Setting s=modelopt(a,t);
 if(x->n==2) {
   K y=getoption(m,t,s);
   return y ? y : knull();
 } else {
   TORCH_CHECK(x->n==3, a,": expecting 3 args, (model; setting; value), ", x->n, " arg(s) supplied");
   return setoption(m,t,s,a,kK(x)[2]), (K)0;
 }
}

static K modelopts(Kmodel *m,K x,bool t,const char *c,SymArrayRef a) {
 if(x->n==2) {
  return getoption(m,t,a);
 } else {
  TORCH_CHECK(x->n==3, c,": given symbols, expecting 3rd argument of values, but given ",x->n," args");
  K y=kK(x)[2];
  TORCH_CHECK(y->t>=0, c,": unable to set values from ",kname(y));
  TORCH_CHECK(y->n == (J)a.size(), c,": ",a.size()," symbol(s) but ",y->n," value(s)");
  setoptions(m,t,c,a,y);
  return (K)0;
 }
}

static void modeldict(Kmodel *m,K x,bool t,const char *c) {
 SymArrayRef a; TORCH_CHECK(xsyms(kK(x)[0],a), c,": unable to get option names from ",kname(x));
 setoptions(m,t,c,a,kK(x)[1]);
}

// ---------------------------------------------------------------------
// modeldata - set data for training/testing model
// modelsetup - set/query options & set data for training/testing model
// ---------------------------------------------------------------------
static int64_t modeldata(Kmodel *m,bool b,const Input& x,const Input& y) {
 return b ? datainit(m->train,m->data,x,y) : datainit(m->test,m->testdata,x,y);
}

static K modeldata(Kmodel *m,K a,bool b,const char *c) {
 Input x,y; std::tie(x,y)=modelargs(a,c);
 return kj(modeldata(m,b,x,y));
}

static K modelsetup(K x,bool t,const char *c) {
 KTRY
  Kmodel *m=xmodel(x);
  if(m) {
   return getoption(m,t,env().alloptions);
  } else {
   m=xmodel(x,0); bool b; S a; SymArrayRef s;
   TORCH_CHECK(m, c,": 1st argument must be a model");
   if(xbool(x,1,b) && x->n==2) {      return getoption(m,t,b);
   } else if(xsym(x,1,a)) {           return modelsym(m,x,t,a);
   } else if(xsyms(x,1,s)) {          return modelopts(m,x,t,c,s);
   } else if(xdict(x,1) && x->n==2) { return modeldict(m,kK(x)[1],t,c), (K)0;
   } else {                           return modeldata(m,x,t,c);
   }
  }
 KCATCH(c);
}

KAPI train(K x) {return modelsetup(x, true,  "train");}
KAPI  test(K x) {return modelsetup(x, false, "test");}

// --------------------------------------------------------------------
// modelinput - [re]set model input(s) or target(s), for train/test
// input/target - [re]set model input(s)/target(s) for training
// testinput/testtarget - [re]set model input(s)/target(s) for testing
// --------------------------------------------------------------------
static K modelinput(K x,bool a,bool b,const char *c) {
 KTRY
  auto *m=xmodel(x);
  if(m) {
   const auto& d=a ? m->data : m->testdata;
   return kin(b ? d.x : d.y);
  } else {
   auto m=xmodel(x,0);
   TORCH_CHECK(m, c,": 1st argument must be a model");
   auto z=modelarg(x,1,c);
   return kj(a ? datainit(m->train,m->data,b,z) : datainit(m->test,m->testdata,b,z));
  }
 KCATCH(c);
}

KAPI      input(K x) {return modelinput(x, true,  true,  "input");}
KAPI     target(K x) {return modelinput(x, true,  false, "target");}
KAPI  testinput(K x) {return modelinput(x, false, true,  "testinput");}
KAPI testtarget(K x) {return modelinput(x, false, false, "testtarget");}

// ----------------------------------------------------------------------------------
// datasize/testsize - return number of inputs in model data with or without batching
// ----------------------------------------------------------------------------------
static auto inputsize(const Input& x) {
 Tensor t;
 if       (auto a=c10::get_if<Tensor>(&x)) {       t=*a;
 } else if(auto a=c10::get_if<TensorVector>(&x)) { if(a->size()) t=a->front();
 } else if(auto a=c10::get_if<TensorDict>(&x))   { if(a->size()) t=a->front().value();
 }
 return t.defined() ? (t.dim() ? t.size(0) : 1) : 0;
}

static K inputsize(K x,bool b,const char *c) {
 KTRY
  auto *m=xmodel(x);
  TORCH_CHECK(m, c,": model argument expected, given ",kname(x));
  return kj(inputsize(b ? m->data.x : m->testdata.x));
 KCATCH(c);
}

KAPI datasize(K x) {return inputsize(x, true,  "datasize");}
KAPI testsize(K x) {return inputsize(x, false, "testsize");}


// ------------------------------------------------------------------------
// modelrun - test/train w'current data or supplying new input(s)/target(s)
// ------------------------------------------------------------------------
static K modelrun(K a,bool b,const char *c) {
 KTRY
  Kmodel *m=xmodel(a);
  if(!m) {
   m=xmodel(a,0);
   TORCH_CHECK(m, c,": model expected as 1st argument, given ",kname(a,0));
   Input x,y; std::tie(x,y)=modelargs(a,c);
   modeldata(m,b,x,y);
  }
  if(b)
   return trainloop(m), getmetrics(m->train,m->data);
  else
   return testloop(m), getmetrics(m->test,m->testdata);
 KCATCH(c);
}

KAPI     run(K x) {return modelrun(x, true,  "run");}
KAPI testrun(K x) {return modelrun(x, false, "testrun");}

// ---------------------------------------------------------------------------
// modeldata - return train/test data as k result or define train/test data
// data/testdata - k api functions to return data to k session
// ---------------------------------------------------------------------------
static K modeldata(K x,bool b,const char *c) {
 KTRY
  auto *m=xmodel(x);
  if(m) {
   const auto& d=b ? m->data : m->testdata;
   return knk(2, kget(d.x), kget(d.y));
  } else {
   m=xmodel(x,0);
   TORCH_CHECK(m, c,": 1st argument must be a model");
   return modeldata(m,x,b,c);
  }
 KCATCH(c);
}

KAPI     data(K x) {return modeldata(x, true,  "data");}
KAPI testdata(K x) {return modeldata(x, false, "testdata");}

// ----------------------------------------------------------------------------
// modelpart - parse args from k to define module, loss & optimizer
// modelkeys - return list of symbols used for model state dictionary
// modelget - return a dictionary with state of module, loss fn & optimizer
// ----------------------------------------------------------------------------
static void modelpart(K x,J i,Kmodule*& q,Kmodule*& l,Kopt*& o) {
 for(;i<x->n;++i) {
  auto* g=xtag(x,i);
  switch(g ? g->a : Class::undefined) {
   case Class::module:    q=g->kmodule();  break;
   case Class::loss:      l=g->kmodule(); break;
   case Class::optimizer: o=(Kopt*)g;  break;
   default: TORCH_ERROR("model arg[",i,"] unrecognized: ",
                    (g ? mapclass(g->a) : kname(x,i))); break;
  }
 }
}

K modelkeys() {
 K x=ktn(KS,5);
 kS(x)[0]=mapclass(Class::module);
 kS(x)[1]=mapclass(Class::loss);
 kS(x)[2]=mapclass(Class::optimizer);
 kS(x)[3]=mapclass(Class::train);
 kS(x)[4]=mapclass(Class::test);
 return x;
}

K modelget(bool a,bool b,Kmodel *m) {
 auto o=m->kopt(); auto l=m->kloss();
 return xD(modelkeys(), 
           knk(5, moduleget(a,b,m->module()),               // module
                  lossget(a,b,l->c,l->module()),            // loss
                  optget(a,b,o->c,o->opt(),o->module()),    // optimizer
                  getoption(m,true,a),                      // training options
                  getoption(m,false,a)));                   // evaluation options
}
 
// ----------------------------------------------------------------------------
// modelfree - free memory from handle to module,loss or optimizer
// model - create model from module, loss & optimizer or retrieve input options
// ----------------------------------------------------------------------------
static void modelfree(K x,J i) {
 for(;i<x->n;++i) 
  TORCH_CHECK(kfree(x,i), "model: unable to free arg[",i,"]");
}

KAPI model(K x) {
 KTRY
  bool a=env().alloptions;
  Kmodule *q=nullptr; Kmodule *l=nullptr; Kopt *o=nullptr; Kmodel *m=nullptr;
  TORCH_CHECK(!x->t, "model not implemented for ",kname(x->t));
  if((m=xmodel(x)) || (x->n==2 && xbool(x,1,a) && (m=xmodel(x,0)))) {
   return modelget(a,false,m);
  } else {
   m=xmodel(x,0); modelpart(x,m ? 1 : 0,q,l,o);
   if(m) {
    if(q) m->q=*q;      //assign new module
    if(l) m->l=*l;      //new loss function
    if(o) m->o=*o;      //new optimizer
    modelfree(x,1);
    return (K)0;
   } else {
    TORCH_CHECK(q && l && o, (q ? (l ? "optimizer" : "loss") : "module")," not specified");
    m=new Kmodel(q,l,o);
    modelfree(x,0);
    return kptr(m);
   }
  }
 KCATCH("model");
}

// ----------------------------------------------------------------------------
// restore - restore tensor/vector/dictionary/model to full size after batching
// testrestore - restore model's test data to full size after batching
// ----------------------------------------------------------------------------
KAPI restore(K x) {
 KTRY
  int64_t d=0,n;
  TORCH_CHECK(!x->t, "restore: not implemented for ",kname(x));
  TORCH_CHECK(0<x->n && x->n<3, "restore: expecting 1-2 args, tensor/vector/dictionary/model & optional dimension, but given ",x->n," args");
  TORCH_CHECK(x->n==1 || xint64(x,1,d), "restore: 2nd arg is dimension, but given ",kname(x,1));
  bool b=x->n==1;
  if(auto a=b ? xten(x) : xten(x,0)) {                       n=fullsize(*a,d);
  } else if (auto a=b ? xvec(x) : xvec(x,0)) {               n=fullsize(*a,d);
  } else if (auto a=b ? xtensordict(x) : xtensordict(x,0)) { n=fullsize(*a,d);
  } else if (auto a=b ? xmodel(x) : xmodel(x,0)) {
   TORCH_CHECK(d==0, "restore: model data is only batched on dimension 0, but given dimension ",d);
   n=fullsize(a->data);
  } else {
   TORCH_ERROR("restore: expecting tensor,vector,dictionary or model, given ",kname(x));
  }
  return kj(n);
 KCATCH("restore")
}

KAPI testrestore(K x) {
 KTRY
  Kmodel *m=xmodel(x);
  TORCH_CHECK(m, "testrestore: not implemented for ",kname(x));
  return kj(fullsize(m->testdata));
 KCATCH("testrestore");
}

// ---------------------------------------------------------------------------
// getdict - translate map of parameter names->values into k dictionary
// ---------------------------------------------------------------------------
static K getdict(const torch::OrderedDict<std::string,bool>& d) {
 size_t i=0; K k=ktn(KS,d.size()), v=ktn(KB,d.size());
 for(const auto& a:d.items())
  kS(k)[i]=cs(a.key().c_str()), kG(v)[i++]=a.value();
 return xD(k,v);
}

static K getdict(const torch::OrderedDict<std::string,c10::optional<double>>& d) {
 size_t i=0; K k=ktn(KS,d.size()), v=ktn(KF,d.size());
 for(const auto& a:d.items())
  kS(k)[i]=cs(a.key().c_str()), kF(v)[i++]=a.value() ? *a.value() : nf;
 return xD(k,v);
}

// ---------------------------------------------------------------------------
// getparms - get copy of parameters from model/module/optimizers
// putparms - put dictionary of parameters into model/module/optimizer
// avgcalc - k interface function to initiate/update/replace averaged weights
// kparms - handle argument parsing for copyparms/avgparms
// copyparms - get/put dictionary of parameters from model/module/optimizer
// avgparms - calculate averages given model/module/optimizer and dictionary
// ---------------------------------------------------------------------------
static TensorDict getparms(const TensorDict& p) {
 TensorDict d;
 for(const auto& t:p.items())
  d.insert(t.key(),t.value().detach().clone());
 return d;
}

static torch::OrderedDict<std::string, bool> putparms(const TensorDict& p,const TensorDict& d) {
 torch::NoGradGuard g;
 torch::OrderedDict<std::string, bool> r;
 for(const auto& t:p.items()) {
  auto a=d.find(t.key());
  if(a)
    t.value().copy_(*a);
  r.insert(t.key(),a);
 }
 return r;
}

static int64_t avgcalc(const TensorDict& p,TensorDict& d) {
 auto *n=d.find(".n");
 if(!n) 
  n=&d.insert(".n",torch::tensor(1.0,torch::kDouble));
 auto f=1.0 / (1.0 + n->item<double>());
 for(const auto& t:p.items()) {
  auto *a=d.find(t.key());
  TORCH_CHECK(a, "avgparms: parameter ",t.key()," not found, unable to compute average");
  a->add_(f * t.value().detach().sub(*a));
 }
 n->add_(1.0);
 return n->item<int64_t>();
}

static K kparms(K x,bool a,const char *c) {
 KTRY
  TORCH_CHECK(!x->t, c," not implemented for ",kname(x));
  TORCH_CHECK(0<x->n && x->n<3, c," expecting 1-2 args, given ",x->n);
  if(x->n==1) {
   auto *g=xtag(x);
   TORCH_CHECK(g, c,": not implemented for ",kname(x));
   TORCH_CHECK(g && (g->a==Class::model || g->a==Class::module || g->a==Class::optimizer),
               c,": expecting model, module or optimizer, given ",kname(x));
   TORCH_CHECK(!a, c,": expects 2 args, model/module/optimizer & dictionary of parameters w'averaged weights");
   return kdict(getparms(g->module().named_parameters())); //detached clones of module parameters
  } else {
   auto *g=xtag(x,0);
   TORCH_CHECK(g && (g->a==Class::model || g->a==Class::module || g->a==Class::optimizer),
               c,": expecting 1st arg of model, module or optimizer, given ",kname(x,0));
   auto *d=xtensordict(x,1);
   TORCH_CHECK(d, c,": expecting dictionary of parameters for 2nd arg, given ",kname(x,1));
   const auto& p=g->module().named_parameters();
   return a ? kj(avgcalc(p,*d)) : getdict(putparms(p,*d));
  }
 KCATCH(c);
}

KAPI copyparms(K x) {return kparms(x, false, "copyparms");}
KAPI  avgparms(K x) {return kparms(x, true,  "avgparms");}

// ---------------------------------------------------------------------------
// stochastic weight averaging requires recalculating batch norm mean & var
// ---------------------------------------------------------------------------
// resetnorm - reset running statistics and return previous momentum setting
// putmomentum - restore previous momentum setting after recalc of statistics
// recalcnorm - recalc batch norm mean & var given model and loaded data
// batchnorm - k api function used to recalculate batch norm statistics 
// ---------------------------------------------------------------------------
using BNMomentum=c10::optional<double>;
using BNMap=torch::OrderedDict<std::string,BNMomentum>;

template<typename M> static BNMomentum resetnorm(M *m) {
 auto r=m->options.momentum();
 m->options.momentum(c10::nullopt); // null implies simple average
 m->reset_running_stats();
 return r; // keep track of original momentum to restore later
}

static BNMap resetnorm(Module& m,const Modulemap& d) {
 BNMap b;
 for(const auto& a:d) {
  if     (auto *m=a.value()->as<torch::nn::BatchNorm1d>()) b.insert(a.key(),resetnorm(m));
  else if(auto *m=a.value()->as<torch::nn::BatchNorm2d>()) b.insert(a.key(),resetnorm(m));
  else if(auto *m=a.value()->as<torch::nn::BatchNorm3d>()) b.insert(a.key(),resetnorm(m));
 }
 return b;
}

static void putmomentum(Module& m,BNMomentum f) {
 if     (auto *bn=m.as<torch::nn::BatchNorm1d>()) bn->options.momentum(f);
 else if(auto *bn=m.as<torch::nn::BatchNorm2d>()) bn->options.momentum(f);
 else if(auto *bn=m.as<torch::nn::BatchNorm3d>()) bn->options.momentum(f);
 else TORCH_ERROR("batchnorm: unrecognized module type, ",m.name());
}

static void putmomentum(Module& m,const Modulemap& p,const BNMap& b) {
 for(const auto& a:b.items()) {
  auto n=p.find(a.key());
  TORCH_CHECK(n, "batchnorm: unable to locate batch norm module '",a.key(),"' to restore momentum");
  putmomentum(**n, a.value());
 }
}

static void putmomentum(Module& m,K x) {
 K k=kK(x)[0], v=kK(x)[1]; auto p=m.named_modules();
 TORCH_CHECK(v->t==KF, "batchnorm: dictionary of previous momentum settings expected as double(s), given ",kname(v));
 for(J i=0; i<k->n; ++i) {
  auto n=p.find(kS(k)[i]); auto f=kF(v)[i];
  TORCH_CHECK(n, "batchnorm: unable to locate batch norm module '",kS(k)[i],"' to restore momentum");
  putmomentum(**n, f==f ? BNMomentum(f) : BNMomentum(c10::nullopt));
 }
}

static void recalcnorm(Kmodel *m,const Modulemap& p,const BNMap& b) {
 auto &d=m->data; const auto& o=m->train; 
 TORCH_CHECK(d.batches()>0, "batchnorm: no training data loaded for the model");
 torch::NoGradGuard g;
 auto *k=m->kmodule(); auto& q=m->module(); bool a=trainflag(q,true);
 while(nextbatch(o,d))
  d.z=mforward(k, o.hidden() && d.batch() > o.task() ? hiddenstate(d) : d.x);
 q.train(a);           // restore previous training mode
 putmomentum(q,p,b);   // restore previous batch norm momentum settings
}

KAPI batchnorm(K x) {
 KTRY
  TORCH_CHECK(!x->t, "batchnorm: not implemented for ",kname(x));
  if(x->n==1) {
   auto *m=xmodel(x); auto *q=xmodule(x);
   TORCH_CHECK(m || q, "batchnorm: expecting model or module, given ",kname(x));
   auto d=(m ? m->module() : q->module()).named_modules(); // all modules
   auto b=resetnorm(m ? m->module() : q->module(), d);     // reset batchnorm modules
   if(m && d.size())
    recalcnorm(m,d,b);
   return getdict(b);
  } else {
   TORCH_CHECK(x->n==2, "batchnorm: expecting 2 args, model/module & dictionary of previous momentum settings, but given ",x->n," args");
   auto *m=xmodel(x,0); auto *q=xmodule(x,0); K y=kK(x)[1];
   TORCH_CHECK(m || q, "batchnorm: expecting model or module for 1st arg, given ",kname(x,0));
   TORCH_CHECK(xdict(y), "batchnorm: expecting k dictionary of momentum settings for 2nd arg, given ",kname(y));
   putmomentum(m ? m->module() : q->module(), y);
   return (K)0;
  }
 KCATCH("batchnorm");
}

// --------------------------------------------------------------------------
// clone - clone module from loss, module, optimizer or model pointer
// --------------------------------------------------------------------------
KAPI Clone(K x) {
 KTRY
  if(auto *a=xten(x)) {
   return kten(a->detach().clone());
  } else if(auto *a=xmodule(x)) {
   return kmodule(a->c, a->m->clone(), a->a);             // clone directly from module
  } else if(auto *a=xmodel(x)) {
   return kmodule(a->kmodule()->c, a->module().clone());  // clone from model pointer
  } else if(auto *a=xoptim(x)) {                          // if optimizer
   return kmodule(mcast(a->m), a->m->clone());            // determine type of module pointer and clone
  } else {
   TORCH_ERROR("clone: expecting module, model or optimizer, not implemented for ",kname(x));
  }
 KCATCH("clone");
}

// ------------------------------------------------------------------------------
// mdict: return dictionary of modules to k
// namelist: vector of strings -> k list of symbols
// nameargs: process args for names/parmnames/buffernames & modules/parms/buffers
// ------------------------------------------------------------------------------
static K mdict(const Modulemap& m) {
 J i=0; K k=ktn(KS,m.size()),v=ktn(0,m.size());
 for(const auto& a:m) {
   kS(k)[i]=cs(a.key().c_str());
   kK(v)[i]=kmodule(mcast(a.value()),a.value());
   ++i;
 }
 return xD(k,v);
}

static K namelist(const std::vector<std::string>& s) {
 K x=ktn(KS,s.size()); size_t i=0;
 for(const auto& a:s)
  kS(x)[i++]=cs(a.c_str());
 return x;
}

static K nameargs(K x,bool a,bool b,State s,const char* c) {
 // a:true to return names, else objects,
 // b:true to return full set of modules, false for immediate children
 KTRY
  K y=nullptr; auto *g=xtag(x); Moduleptr m;
  if(!g) {
   g=xtag(x,0);
   TORCH_CHECK(g, c,": expecting pytorch object, not implemented for ",kname(x));
   TORCH_CHECK(!x->t && x->n==2, c,": expecting up to two args, (object; index/name), given ",x->n," args");
   y=kK(x)[1];
   TORCH_CHECK(y->t==-KJ || y->t==-KS, c,": expecting integer or symbol for 2nd arg, given ",kname(x,1));
  }
  switch(g->a) {
   case Class::dict: TORCH_CHECK(a,  c,": not implemented for dictionary");
                     TORCH_CHECK(!y, c,": dictionary supplied with extraneous argument");
                     return namelist(g->dict().keys());
   case Class::loss:
   case Class::module:
   case Class::model:
   case Class::optimizer: m=g->moduleptr(); break;
   default: TORCH_ERROR(c,": not implemented for ",mapclass(g->a));
  }
  if(y) {
   if(y->t == -KJ) {
    const auto& v=m->modules(false); J n=v.size();
    TORCH_CHECK(-1<y->j && y->j < n, c,": invalid index[",y->j,"] for ",n," child module",(n==1 ? "" : "s"));
    m=v.at(y->j);
   } else {
    m=m->named_modules()[y->s];
   }
  }
  if(a) {
   switch(s) {
    case State::module:  return namelist(b ? m->named_modules("",false).keys() : m->named_children().keys());
    case State::parms:   return namelist(m->named_parameters().keys());
    case State::buffers: return namelist(m->named_buffers().keys());
    default: TORCH_ERROR(c,": unrecognized mode");
   }
  } else {
   switch(s) {
    case State::module:  return mdict(b ? m->named_modules("",false) : m->named_children());
    case State::parms:   return kdict(m->named_parameters(),Cast::parameter);
    case State::buffers: return kdict(m->named_buffers(),Cast::buffer);
    default: TORCH_ERROR(c,": unrecognized mode");
   }
  }
 KCATCH(c);
}

KAPI       names(K x) {return nameargs(x, true,  true,  State::module,  "names");}
KAPI  childnames(K x) {return nameargs(x, true,  false, State::module,  "children");}
KAPI   parmnames(K x) {return nameargs(x, true,  true,  State::parms,   "parmnames");}
KAPI buffernames(K x) {return nameargs(x, true,  true,  State::buffers, "buffernames");}

KAPI     modules(K x) {return nameargs(x, false, true,  State::module,  "modules");}
KAPI    children(K x) {return nameargs(x, false, false, State::module,  "children");}
KAPI       parms(K x) {return nameargs(x, false, true,  State::parms,   "parms");}
KAPI     buffers(K x) {return nameargs(x, false, true,  State::buffers, "buffers");}

// -----------------------------------------------------------------------------------------
//  child - k api function, expects model/module/optimizer and name, returns submodule
// -----------------------------------------------------------------------------------------
KAPI child(K x) {
 KTRY
  Ktag *g=xtag(x,0);
  TORCH_CHECK(g, "child: expects module/model/optimizer as first argument");
  TORCH_CHECK(x->n==2, "child: expects 2 args, (module/model/optimizer; name/index), ",x->n," arg(s) given");
  K y=kK(x)[1]; Moduleptr m;
  TORCH_CHECK(y->t == -KS || y->t == -KJ, "child: expects 2nd arg of name or index, given ",kname(x,1));
  if(y->t == -KJ) {
   const auto& v=g->module().modules(false); J n=v.size();
   TORCH_CHECK(-1<y->j && y->j < n, "child: invalid index[",y->j,"] for ",n," child module",(n==1 ? "" : "s"));
   m=v.at(y->j);
  } else {
   m=strchr((C*)y->s,'.') ? g->module().named_modules()[y->s] : g->module().named_children()[y->s];
  }
  return kmodule(mcast(m),m);
 KCATCH("child");
}

// -----------------------------------------------------------------------------
// moduletypes - given module, return dictionary of name -> module type
// accepts 2nd arg of type(s), returning only module names matching type(s)
// -----------------------------------------------------------------------------
KAPI moduletypes(K x) {
 KTRY
  auto *g=xtag(x); if(!g) g=xtag(x,0);
  TORCH_CHECK(g, "moduletypes: 1st argument is not a recognizable pytorch object");
  const auto& m=g->module().named_modules("",true);
  if(x->n==1) {
   J i=0; K k=ktn(KS,m.size()),v=ktn(KS,m.size());
   for(const auto& a:m)
    kS(k)[i]=cs(a.key().c_str()), kS(v)[i++]=msym(mcast(a.value()));
   return xD(k,v);
  } else if(x->n==2) {
   SymArrayRef s;
   TORCH_CHECK(xsyms(x,1,s), "moduletypes: expecting 2nd arg of module types(symbols), given ",kname(x,1));
   bool b=kK(x)[1]->t == KS;      //true if checking for a list of module types
   K k=ktn(KS,0); K v=b ? ktn(KS,0) : nullptr;
   for(const auto& i:s) msym(i);  //check that given module type(s) are recognized
   for(const auto& a:m) {
    S c=msym(mcast(a.value()));
    for(const auto& i:s) {
     if(c==i) {
      js(&k,cs(a.key().c_str()));
      if(b) js(&v,c);
     }
    }
   }
   return b ? xD(k,v) : k;
  } else {
   TORCH_ERROR("moduletypes: expecting 1-2 arguments, e.g. module or (module;types), but ",x->n," args supplied");
  }
 KCATCH("types");
}

// -----------------------------------------------------------------------
// join_name - make nested name, e.g. parent.child.subchild
// parmtypes - given module, return dictionary of parm name -> module type
// -----------------------------------------------------------------------
static std::string join_name(const std::string& x, const std::string& y) {
  size_t n = y.size();
  if(!x.empty())
    n += x.size() + 1;
  std::string s; s.reserve(n);
  if(!x.empty()) {
    s += x;
    s.push_back('.');
  }
  s += y;
  return s;
}

KAPI parmtypes(K x) {
 KTRY
  const auto *g=xtag(x);
  TORCH_CHECK(g, "parmtypes: expecting model, module or optimizer, given ",kname(x));
  const auto& m=g->module();
  K k=ktn(KS,0),v=ktn(KS,0);
  m.apply([&](const std::string& s,const Module& m) {
   const auto& p=access_private::parameters_(m);
   if(p.size()) {
    S c=msym(mcast(m));
    for(const auto& a:p)
     if(a.value().defined())
      js(&k,cs(join_name(s,a.key()).c_str())), js(&v,c);
    }
  });
  return xD(k,v);
 KCATCH("parmtypes");
}


// ----------------------------------------------------------------------
// tensorarg - process args for getting/setting a module parameter/buffer
// parm/buffer - a api functions to retrieve or reset parameter/buffer
// ----------------------------------------------------------------------
static K tensorarg(K x,Cast c,const char *e) {
 KTRY
  auto *g=xtag(x,0); S s; Moduleptr m;
  TORCH_CHECK(g, e,": first arg must be a module, model, loss or optimizer");
  TORCH_CHECK(x->n>1, e,": expecting 2nd argument of ",tensortype(c)," name");
  TORCH_CHECK(x->n<4, e,": expecting 2-3 args, e.g. (module;",tensortype(c)," name) or (module;",tensortype(c)," name;value), but given ",x->n," args");
  TORCH_CHECK(xsym(x,1,s), e,": second arg is ",tensortype(c)," name, expecting symbol given ",kname(x,1));
  switch(g->a) {
   case Class::loss:
   case Class::module:
   case Class::model:
   case Class::optimizer: m=g->moduleptr(); break;
   default: TORCH_ERROR(e,": not implemented for ",mapclass(g->a));
  }
  const auto *a=findtensor(*m,s,c);
  if(x->n==2) {
   TORCH_CHECK(a, e,": ",tensortype(c)," `",s," not found");
   return kten(*a);
  } else {
   auto *t=xten(x,2);
   if(a) {
    torch::NoGradGuard ng;
    a->copy_(t ? *t : kput(x,2));
   } else if(c==Cast::parameter) {
    m->register_parameter(s, t ? *t : kput(x,2).to(firstdevice(g)));
   } else if(c==Cast::buffer) {
    m->register_buffer(s, t ? *t : kput(x,2).to(firstdevice(g)));
   } else {
    TORCH_ERROR("unrecognized tensor type, cannot set parameter/buffer");
   }
   return (K)0;
  } 
 KCATCH(e);
}

KAPI   parm(K x) { return tensorarg(x, Cast::parameter, "parm"); }
KAPI buffer(K x) { return tensorarg(x, Cast::buffer,    "buffer"); }

// -------------------------------------------------------------------------------------------
// kfreeze - handle k arguments and resultant namelists/vector/dictionary
// freeze,unfreeze - given module, parameter name(s), unset/set gradient and optional value(s)
// -------------------------------------------------------------------------------------------
static void kfreeze(bool g,const char *c,Ktag *k) {
 switch(k->a) {
  case Class::tensor: k->tensor().set_requires_grad(g); break;
  case Class::vector: for(auto& t:k->vector()) t.set_requires_grad(g); break;
  case Class::dict:   for(auto& i:k->dict().items()) i.value().set_requires_grad(g); break;
  case Class::module:
  case Class::model:  for(auto& t:k->module().parameters()) t.set_requires_grad(g); break;
  default: TORCH_ERROR(c,": not implemented for ",mapclass(k->a)); break;
 }
}

static void kfreeze(bool g,const char *c,const TensorDict& p,const SymArrayRef& s,const TensorVector& v) {
 torch::NoGradGuard nograd;
 size_t i=0;
 for(i=0; i<s.size(); ++i) {  // 1st pass to verify names & sizes (if value(s) supplied)
  auto *x=p.find(s[i]);
  TORCH_CHECK(x, c,": unable to find parameter `",s[i]);
  if(v.size()) {
   const auto& y=v.at(i);
   TORCH_CHECK(broadcast(*x,y), c,": size mismatch for parameter `",s[i],", size is ",x->sizes()," vs new size of ",y.sizes());
  }
 }
 for(i=0; i<s.size(); ++i) {  // 2nd pass to set grad/nograd and values if any
  auto *x=p.find(s[i]);
  TORCH_CHECK(x, c,": unable to find parameter `",s[i]);
  x->set_requires_grad(g);
  if(v.size()) 
   x->copy_(v.at(i));
 }
}

static void kfreeze(bool g,const char *c,const TensorDict& p,const TensorDict& d) {
 torch::NoGradGuard nograd;
 for(const auto& a:d) { // 1st pass to check names and sizes are all ok
  auto *x=p.find(a.key());
  TORCH_CHECK(x, c,": unable to find parameter `",a.key());
  TORCH_CHECK(broadcast(*x,a.value()),
              c,": size mismatch for parameter `",a.key(),", size is ",x->sizes(),
                " vs new size of ",a.value().sizes());
 }
 for(const auto& a:d) { // 2nd pass updates gradient flag and values
  auto *x=p.find(a.key());
  TORCH_CHECK(x, c,": unable to find parameter `",a.key());
  x->set_requires_grad(g); x->copy_(a.value());
 }
}

static K kfreeze(K x,bool g,const char *c) {
 KTRY
  if(auto *a=xtag(x)) {
   kfreeze(g,c,a);
  } else {
   TORCH_CHECK(!x->t, c,": not implemented for ",kname(x));
   TORCH_CHECK(1<x->n && x->n < 4, c,": expecting 2-3 args, (module;names;values), ",x->n," given");
   auto *m=xmodel(x,0); auto *q=m ? m->kmodule() : xmodule(x,0); auto *d=q ? nullptr : xtensordict(x,0);
   TORCH_CHECK(q || d, c,": expecting 1st arg of module,model or parameter dictionary, given ",kname(x,0));
   SymArrayRef s; const auto& p=d ? *d : q->module().named_parameters();
   if(xsyms(x,1,s)) {
    if(x->n==2) {
     kfreeze(g,c,p,s,{});
    } else {
     Input in=Empty(); modelarg(kK(x)[2],c,nullptr,in);
     if(xsym(x,1)) {
      auto *v=c10::get_if<Tensor>(&in);
      TORCH_CHECK(v, c,": given single parameter name, expected single tensor or array value, given ",inputname(in));
      kfreeze(g,c,p,s,{*v});
     } else {
      auto *v=c10::get_if<TensorVector>(&in);
      TORCH_CHECK(v, c,": given parameter list, expected vector of tensor values, given ",inputname(in));
      TORCH_CHECK(s.size()==v->size(), c,": ",s.size(),"-element list of names but ",v->size(),"-element list of values");
      kfreeze(g,c,p,s,*v);
     }
    }
   } else {
    TORCH_CHECK(x->n==2, c,": expecting 2nd arg of dictionary, but given ",x->n," args");
    if (xdict(x,1)) {
     const auto& d=kputd(kK(x)[1]);
     kfreeze(g,c,p,d);
    } else if(auto *d=xtensordict(x,1)) {
     kfreeze(g,c,p,*d);
    } else {
     TORCH_ERROR(c,": unrecognized 2nd arg, ",kname(x,1));
    }
   }
  }
  return (K)0;
 KCATCH(c);
}

KAPI   freeze(K x) {return kfreeze(x, false, "freeze");}
KAPI unfreeze(K x) {return kfreeze(x, true,  "unfreeze");}

// ----------------------------------------------
// add model api functions to library dictionary
// ----------------------------------------------
void modelfn(K x) {
 fn(x, "avgparms",    KFN(avgparms),    1);
 fn(x, "backstep",    KFN(backstep),    1);
 fn(x, "backward",    KFN(backward),    1);
 fn(x, "batchinit",   KFN(Batchinit),   1);
 fn(x, "batch",       KFN(trainbatch),  1);
 fn(x, "batchnorm",   KFN(batchnorm),   1);
 fn(x, "buffer",      KFN(buffer),      1);
 fn(x, "buffernames", KFN(buffernames), 1);
 fn(x, "buffers",     KFN(buffers),     1);
 fn(x, "child",       KFN(child),       1);
 fn(x, "childnames",  KFN(childnames),  1);
 fn(x, "children",    KFN(children),    1);
 fn(x, "clip",        KFN(clip),        1);
 fn(x, "clipv",       KFN(clipv),       1);
 fn(x, "clone",       KFN(Clone),       1);
 fn(x, "copyparms",   KFN(copyparms),   1);
 fn(x, "data",        KFN(data),        1);
 fn(x, "datasize",    KFN(datasize),    1);
 fn(x, "eforward",    KFN(eforward),    1);
 fn(x, "evaluate",    KFN(evaluate),    1);
 fn(x, "forward",     KFN(forward),     1);
 fn(x, "freeze",      KFN(freeze),      1);
 fn(x, "input",       KFN(input),       1);
 fn(x, "kforward",    KFN(kforward),    1);
 fn(x, "model",       KFN(model),       1);
 fn(x, "modules",     KFN(modules),     1);
 fn(x, "moduletypes", KFN(moduletypes), 1);
 fn(x, "names",       KFN(names),       1);
 fn(x, "nforward",    KFN(nforward),    1);
 fn(x, "nograd",      KFN(nograd),      1);
 fn(x, "parm",        KFN(parm),        1);
 fn(x, "parmnames",   KFN(parmnames),   1);
 fn(x, "parms",       KFN(parms),       1);
 fn(x, "parmtypes",   KFN(parmtypes),   1);
 fn(x, "restore",     KFN(restore),     1);
 fn(x, "run",         KFN(run),         1);
 fn(x, "shuffle",     KFN(shuffle),     1);
 fn(x, "target",      KFN(target),      1);
 fn(x, "testbatch",   KFN(testbatch),   1);
 fn(x, "testdata",    KFN(testdata),    1);
 fn(x, "testinit",    KFN(testinit),    1);
 fn(x, "testinput",   KFN(testinput),   1);
 fn(x, "test",        KFN(test),        1);
 fn(x, "testsize",    KFN(testsize),    1);
 fn(x, "testrestore", KFN(testrestore), 1);
 fn(x, "testrun",     KFN(testrun),     1);
 fn(x, "testtarget",  KFN(testtarget),  1);
 fn(x, "training",    KFN(training),    1);
 fn(x, "train",       KFN(train),       1);
 fn(x, "unfreeze",    KFN(unfreeze),    1);
 fn(x, "unshuffle",   KFN(unshuffle),   1);
 fn(x, "zerograd",    KFN(zerograd),    1);
}
