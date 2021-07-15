#include "ktorch.h"

// ----------------------------------------------------------------------------
// modelpart - parse args from k to define module, loss & optimizer
// modelkeys - return list of symbols used for model state dictionary
// modelstate - return a dictionary with state of module, loss fn & optimizer
// model - create model from module, loss & optimizer or retrieve input options
// ----------------------------------------------------------------------------
static void modelpart(K x,J i,Kmodule*& q,Kmodule*& l,Kopt*& o) {
 for(;i<x->n;++i) {
  auto* g=xtag(x,i);
  switch(g ? g->a : Class::undefined) {
   case Class::module:    q=(Kmodule*)g;  break;
   case Class::loss:      l=(Kmodule*)g; break;
   case Class::optimizer: o=(Kopt*)g;  break;
   default: TORCH_ERROR("model arg[",i,"] unrecognized: ",
                    (g ? mapclass(g->a) : kname(x,i))); break;
  }
 }
}

K modelkeys() {
 K x=ktn(KS,3);
 kS(x)[0]=mapclass(Class::module);
 kS(x)[1]=mapclass(Class::loss);
 kS(x)[2]=mapclass(Class::optimizer);
 return x;
}

K modelstate(bool a,bool b,Kmodel *m) {
 return xD(modelkeys(), knk(3, mget(a,b,*m->m), lossdict(a,b,m->lc,*m->l), optstate(a,b,m)));
}

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
   return modelstate(a,false,m);
  } else {
   m=xmodel(x,0); modelpart(x,m ? 1 : 0,q,l,o);
   if(m) {
    if(q) m->mc=q->c, m->m=q->m;             //assign new module
    if(l) m->lc=l->c, m->l=l->m;             //new loss function
    if(o) m->oc=o->c, m->o=o->o, m->om=o->m; //new optimizer
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

// -----------------------------------------------------------------------------------------
// zerograd - zero gradients on tensor, vector of tensors, optimizer, sequential or model
// -----------------------------------------------------------------------------------------
KAPI zerograd(K x) {
 KTRY
  auto *g=xtag(x);
  auto f=[](Tensor& t) { if(t.grad().defined()){t.grad().detach_(); t.grad().zero_();} };
  TORCH_CHECK(g, "zerograd not implemented for ",kname(x->t));
  switch(g->a) {
   case Class::tensor:     f(((Kten*)g)->t); break;
   case Class::vector:     for(auto& t:((Kvec*)g)->v) f(t); break;
   case Class::module:     ((Kmodule*)g)->m->zero_grad(); break;
   case Class::optimizer:  ((Kopt*)g)->o->zero_grad(); break;
   case Class::model:      ((Kmodel*)g)->o->zero_grad(); break;
   default: TORCH_ERROR("zerograd not implemented for ",mapclass(g->a));
  }
  return (K)0;
 KCATCH("zero gradients");
}

// -------------------------------------------------------------------------------------------
// kforward - return tensor from parsing k args & running forward calcs on input(s)
// forward - forward calcs on module/model given inputs & targets
// -------------------------------------------------------------------------------------------
static K kforward(Cast c,Module& m,K a) {
 Tensor x,y,z; TensorVector *v;
 if((v=xvec(a,1))) {
  TORCH_CHECK(v->size(), "forward: empty vector of tensors supplied");
  IntArrayRef i;
  if(a->n==2) {
   return kout(mforward(c, m, v->at(0)));
  } else if(a->n==3 && xsize(a,2,i)) {
   switch(i.size()) {
    case 1: return kout(mforward(c, m, v->at(i[0])));
    case 2: return kout(mforward(c, m, v->at(i[0]), v->at(i[1])));
    case 3: return kout(mforward(c, m, v->at(i[0]), v->at(i[1]), v->at(i[2])));
    default: TORCH_ERROR("forward: vector w'indices expects 1-3 indices, ",i.size()," supplied");
   }
  } else {
   TORCH_ERROR("forward with vector expects format of (module/model;vector) or (module/model;vector;indices)");
  }
 } else {
  TORCH_CHECK(!a->t && a->n>1 && a->n<5, "forward expects 2-4 args: module/model and up to 3 tensors/arrays, e.g. (m;x) or (m;x;y;z)");
  if(!xten(a,1,x))            x=kput(a,1);
  if(a->n>=3 && !xten(a,2,y)) y=kput(a,2);
  if(a->n==4 && !xten(a,3,z)) z=kput(a,3);
  return kout(a->n==2 ? mforward(c,m,x) : (a->n==3 ? mforward(c,m,x,y) : mforward(c,m,x,y,z)));
 }
}

KAPI forward(K x) {
 KTRY
  Ktag *g=xtag(x,0);
  TORCH_CHECK(g, "forward: expects module/model as first arg, with tensor(s)/vector/dictionary as additional args");
  switch(g->a) {
   case Class::module: {auto *m=(Kmodule*)g; return kforward(m->c, *m->m,x);}
   case Class::model:  {auto *m=(Kmodel*)g;  return kforward(m->mc,*m->m,x);}
   default: TORCH_ERROR("forward not implemented for ",mapclass(g->a));
  }
 KCATCH("forward");
}

// -------------------------------------------------------------------------------------------
// losstensor - given module output, return output tensor to use for loss calc
// mloss - given model and vector of inputs, e.g. v=x,y, loss=loss(module(v[0]),v[1])
// mbackward - given model, input & target, do forward calcs, get loss, backward prop on loss
// -------------------------------------------------------------------------------------------
static Tensor losstensor(Cast c,const Output& o) {
 switch(c) {
  case Cast::cosineloss:
  case Cast::margin:
  case Cast::triplet:
  case Cast::ctc:
   TORCH_ERROR("unable to get tensor for loss calculation, not implemented for ",lmap(c));
  default:
   if(auto a=c10::get_if<Tensor>(&o)) {
    return *a;
   } else if(auto a=c10::get_if<Tuple>(&o)) {
    return std::get<0>(*a);
   } else if(auto a=c10::get_if<Tensors>(&o)) {
    return a->front();
   } else if(auto a=c10::get_if<TensorVector>(&o)) {
    return a->front();
   } else {
    TORCH_ERROR("unable to get tensor for loss calculation -- unrecognized module output");
   }
 }
}

Tensor mloss(Kmodel *m,const Tensor& x,const TensorVector &v) {
 if(v.size()==2)
  return lossfwd(m->lc,*m->l,x,v[1]);
 else if(v.size()==3)
  return lossfwd(m->lc,*m->l,x,v[1],v[2]);
 else
  TORCH_ERROR("model: ", v.size()," inputs given, expecting 2-3");
}

Tensor mloss(Kmodel *m,const TensorVector &v) {
 return mloss(m, losstensor(m->lc, mforward(m->mc,*m->m,v.at(0))), v);
}

Tensor mloss(Kmodel *m,const Tensor& x,const Tensor& y) {
 return lossfwd(m->lc,*m->l, losstensor(m->lc, mforward(m->mc,*m->m,x)), y);
}

K mbackward(K a) {
 Kmodel *m=xmodel(a,0); Tensor *x,*y,r; TensorVector *v;
 TORCH_CHECK(m, "backward: first argument not a model");
 if((x=xten(a,1)) && (y=xten(a,2)) && a->n==3) {
  r=mloss(m,*x,*y);
 } else if ((v=xvec(a,1)) && a->n==2) {
  TORCH_CHECK(v->size()>1, "backward: vector expected to contain two or more tensors, given ",v->size());
  r=mloss(m,*v);
 } else {
  TORCH_ERROR("backward expects (model; inputs; targets) or (model;vector)");
 }
 r.backward();
 return kget(r);
}

// -----------------------------------------------------------------------------------------
// tbackward - backprop given tensor, optional tensor & sym for retain/create gradient graph
// kbackward - backward calcs on tensor or model(uses model loss(model output,target) )
// -----------------------------------------------------------------------------------------
static K tbackward(K x) {
 Tensor t; bool ok=false;
 if(xten(x,t)) {
  t.backward(); ok=true;
 } else if(xten(x,0,t)) {
  bool a=false,b=false; Tensor g; J n=x->n - xbacksym(x,x->n-1,a,b);
  if(n==1) {
    t.backward({},a,b); ok=true;
  } else if(n==2) {
   if(!xten(x,1,g)) g=kput(x,1).to(t.device());
   if(!g.dim() && t.dim()) g.resize_as_(t).fill_(g[0]);
   t.backward(g,a,b); ok=true;
  } else if(n==1) {
    t.backward({},a,b); ok=true;
  }
 }
 TORCH_CHECK(ok, "backward: unexpected arg(s), expecting tensor, (tensor;sym), (tensor;grad tensor/array) or (tensor;grad tensor/array;sym)");
 return (K)0;
}

KAPI kbackward(K x) {
 KTRY
  Ktag *g;
  TORCH_CHECK((g=xtag(x)) || (g=xtag(x,0)), "backward expects a tensor or model as first arg");
  switch(g->a) {
   case Class::tensor: return tbackward(x);
   case Class::model:  return mbackward(x);
   default: TORCH_ERROR("backward not implemented for ",mapclass(g->a));
  }
  return (K)0;
 KCATCH("backward");
}

// -------------------------------------------------------------------------------------------
// trainbatch - run model's forward calc, loss, backward calcs and optimizer step in batches
// train - train model for given batch size and number of passes through the data ("epochs")
// ktrain - k api fn, expects (model;vector;batch size; optional epochs;optional shuffle flag)
// -------------------------------------------------------------------------------------------
Tensor trainbatch(Kmodel *m,TensorVector& v,int64_t w,int64_t n=0);
Tensor trainbatch(Kmodel *m,TensorVector& v,int64_t w,int64_t n) {
 auto *o=m->o.get();

 if(!n) n=maxsize(v);
 if(w>n) w=n;                          // reduce batch size if exceeds total size
 auto s=subsets(w,n);                  // no. of subsets to process
 auto r=torch::empty(s);               // tensor for batch losses
 auto* p=r.data_ptr<float>();          // ptr for quicker assignment

 auto loss=[&]() {                     // define closure for
  m->o->zero_grad();                   // resetting gradients
  auto r=mloss(m,v);                   // calculating model output & loss
  r.backward();                        // calculating gradients
  return r;                            // return loss tensor
 };

 for(int64_t i=0,j=0; i<n; i+=w,++j) {
  subset(v,0,i,w,n);                   // narrow tensors to current batch
  if(m->oc == Cast::lbfgs)
   p[j]=o->step(loss).item<float>();     // pass closure, e.g. LBFGS
  else
   p[j]=loss().item<float>(), o->step(); // single loss evaluation
 }
 subset(v,0,0,n,n);                    // reset tensors to full length
 return r;                             // return losses
}

Tensor train(Kmodel *m,TensorVector& v,int64_t w,int64_t e,bool s) {
 auto n=fullsize(v);
 if(e) {
  TensorVector r;
  for(int64_t i=0; i<e; ++i) {
   if(s) shuffle_(v);
   r.emplace_back(trainbatch(m,v,w,n));
  }
  return torch::stack(r);
 } else {
  if(s) shuffle_(v);
  return trainbatch(m,v,w,n);
 }
}

KAPI ktrain(K x) {
 KTRY
  Kmodel *m; TensorVector *v; bool s=true; int64_t w,e=0;
  TORCH_CHECK(!x->t, "train: not implemented for ",kname(x->t));
  auto a=x->n - xbool(x,x->n-1,s);
  if((m=xmodel(x,0)) && (v=xvec(x,1)) && xint64(x,2,w) && (a==3 || (a==4 && xint64(x,3,e)))) {
   TORCH_CHECK(w>0,  "train: batch size must be positive");
   TORCH_CHECK(e>-1, "train: number of passes cannot be negative");
   TORCH_CHECK(v->size(), "train: vector of inputs is empty");
   return kget(train(m,*v,w,e,s));
  } else {
   return KERR("train: unrecognized arg(s)");
  }
 KCATCH("train");
}

// --------------------------------------------------------------------------------------------
// evalfwd - forward calc on given module layer and inputs, in batches if batchsize given
// --------------------------------------------------------------------------------------------
static Tensor evalfwd(Cast c,Module& m,Tensor& x) {
 auto r=mforward(c,m,x); auto a=c10::get_if<Tensor>(&r);
 TORCH_CHECK(a, "evaluate: ",msym(c)," output not simple tensor, not implemented");
 return *a;
}

static Tensor evalfwd(Cast c,Module& m,Tensor& x,int64_t w) {
 bool b=m.is_training(); Tensor y;
 if(b) m.train(false);               // turn off training mode
 if(w) {                             // if batches of window size w
  auto n=maxsize(x);                 // get maxsize
  TensorVector r;
  for(int64_t i=0; i<n; i+=w) {      // batches of w
   subset(x,0,i,w,n);
   r.emplace_back(evalfwd(c,m,x));  // accumulate forward calcs
  }
  subset(x,0,0,n,n);                 // restore size of inputs
  y=torch::cat(r);                   // and join batch results
 } else {
  y=evalfwd(c,m,x);                  // no batching, run forward on full input
 }
 if(b) m.train(true);
 return y;
}

// --------------------------------------------------------------------------------------------
//  metric  - map k symbol to metric, e.g. `accuracy -> Metric::accuracy
//          - calculate single metric given model, vector of inputs/targets, tensor of fwd calc
//  metrics - handle multiple metrics, return scalar/list/tensor or k list of metrics
// --------------------------------------------------------------------------------------------
static Metric metric(S s) {
 for(const auto& m:env().metric) 
  if(std::get<0>(m)==s) return std::get<1>(m);
 TORCH_ERROR("unrecognized metric: ",s);
}

static Tensor metric(Metric e,Kmodel *m,const TensorVector& v,const Tensor& y) {
 switch(e) {
  case Metric::accuracy:  TORCH_CHECK(v.size()>=2, "accuracy metric: no target found");
                          return 100.0*torch::sum(v[1].eq(y.argmax(-1)))/y.size(0);
  case Metric::loss:      TORCH_CHECK(v.size()>=2, "loss metric: no target found");
                          TORCH_CHECK(m,"loss metric: unable to determine loss function");
                          return mloss(m,y,v);
  case Metric::max:       return torch::argmax(y,-1);
  case Metric::out:       return y;
  default: TORCH_ERROR("unrecognized metric");
 }
}

static K metrics(Kmodule *q,Kmodel *m,TensorVector& v,int64_t w,bool b,K s) {
 Tensor y=evalfwd(q ? q->c : m->mc,q ? *q->m : *m->m, v[0], w);
 if(s) {
  if(s->t == -KS) {
   return kresult(b, metric(metric(s->s),m,v,y));
  } else {
   K x=ktn(0, s->n);
   try {
    for(J i=0; i<s->n; ++i)
     kK(x)[i]=kresult(b, metric(metric(kS(s)[i]),m,v,y));
   } catch(...) {
    if(x) r0(x);
    throw;
   }
   return x;
  }
 } else {
  return kresult(b, y);
 }
}

KAPI evaluate(K x) {
 KTRY
  torch::NoGradGuard g;
  Kmodule *q=xmodule(x,0); Kmodel *m=xmodel(x,0); TensorVector *v; bool b=false; int64_t w=0;
  TORCH_CHECK(q || m, "evaluate: expects (model/module; vector/tensor(s)/array(s);optional args..)\n"
                      "          optional args: (batch size; tensor flag; metric(s))");
  J n=x->n; K s=nullptr;
  if(abs(kK(x)[n-1]->t)==KS) n--, s=kK(x)[n];  // metric symbol(s) given as last arg
  if(n>2 && xbool(x,n-1,b)) n--;               // tensor flag at the end of remaining args
  if(n>2 && xint64(x,n-1,w)) n--;              // batch size at the end of remaining args
  TORCH_CHECK(n>1, "evaluate: expects at least one input as 2nd arg");
  if((v=xvec(x,1))) {
   return metrics(q,m,*v,w,b,s);
  } else {
   TensorVector a;
   for(J i=1;i<n;++i) {Tensor* t=xten(x,i); a.emplace_back(t ? *t : kput(x,i));}
   return metrics(q,m,a,w,b,s);
  }
 KCATCH("evaluate");
}

// -------------------------------------------------------------------------------------------
// training - query/set training flag given model or module layer
// -------------------------------------------------------------------------------------------
KAPI training(K x) {
 KTRY
  bool b; Ktag *g;
  TORCH_CHECK((g=xtag(x)) || ((g=xtag(x,0)) && x->n==2 && xbool(x,1,b)),
              "training: unrecognized arg(s), expects module/model and optional flag");
  TORCH_CHECK(g->a==Class::module || g->a==Class::model, "training: not implemented for ",mapclass(g->a));
  auto& m=g->a==Class::module ? ((Kmodule*)g)->m : ((Kmodel*)g)->m;
  return (x->n==2) ? m->train(b),(K)0 : kb(m->is_training());
 KCATCH("training");
}

// ------------------------------------------------------------------------------------------
// kclip - handle input ptr to allocated tensor/vector/dict/module/model and args
//  clip - api function for clipping gradient norm
// clipv - api function for clipping gradient value
// ------------------------------------------------------------------------------------------
static F kclip(bool a,F f,F p,const TensorVector& v) {
 if(a)
  return torch::nn::utils::clip_grad_norm_(v,f,p);
 else
  return torch::nn::utils::clip_grad_value_(v,f), 0;
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
  if(x->n>2)
   TORCH_CHECK(xnum(x,2,p), c,": 3rd arg, norm exponent, is ",kname(x,2),", expecting long/double");
  if(x->n>3)
   TORCH_CHECK(xbool(x,3,b), c,": 4th arg, group flag, is ",kname(x,3),", expecting boolean");
  switch(g->a) {
   case Class::tensor:    return kf(kclip(a,f,p,TensorVector{((Kten*)g)->t}));
   case Class::vector:    return kf(kclip(a,f,p,((Kvec*)g)->v));
   case Class::dict:      return kf(kclip(a,f,p,((Kdict*)g)->d.values()));
   case Class::module:    return kf(kclip(a,f,p,((Kmodule*)g)->m->parameters()));
   case Class::model:     return kf(kclip(a,f,p,((Kmodel*)g)->m->parameters()));
   case Class::optimizer:
    if(a || !b) { //if clip by value or group flag off, consider all parameters across groups for clipping
     return kf(kclip(a,f,p,((Kopt*)g)->m->parameters()));
    } else {
     const auto& y=((Kopt*)g)->o->param_groups();
     K r=ktn(KF,y.size()); J i=0;
     for(const auto& z:y) kF(r)[i++]=kclip(a,f,p,z.params()); //clip by parameter group
     return r;
    }
   default: TORCH_ERROR(c,": not implemented for ",mapclass(g->a));
  }
 KCATCH(c);
}

KAPI clip(K x)  {return kclip(x, true,  "clip gradient norm");}
KAPI clipv(K x) {return kclip(x, false, "clip gradient value");}

// ----------------------------------------------------------------------------------------
// mvec - routines to index into vector of tensors as part of args to forward/backward etc
// ----------------------------------------------------------------------------------------
static void mvec(bool b,const char *c,K x,Tensors& t,TensorVector *v) {
 // attempt to set tensor(s) in array t with index/indices into vector v from input x
 if(x->t == -KJ) {
  t[0]=v->at(x->j);
 } else if(x->t == KJ) {
  TORCH_CHECK(x->n>0 && x->n<4, c,": expecting 1-3 ",(b ? "target" : "input")," indices, given ",x->n);
  for(J i=0; i<x->n; ++i)
   t[i]=v->at(kJ(x)[i]);
 } else {
  TORCH_ERROR(c,": ",(b ? "target" : "input")," indices expected, given ",kname(x));
 }
}

static void mvec(const char *c,K a,Tensors& x,Tensors& y,TensorVector *v) {
 // attempt to set tensor(s) in arrays x,y with indices into vector v from input args a
 if(a->n==2) {
  TORCH_CHECK(v->size()==2, c,": unable to determine input & target given vector of ",v->size()," elements");
  x[0]=v->at(0); y[0]=v->at(1);
 } else {
  TORCH_CHECK(a->n==3, c,": expecting 3 arguments, (model;vector;indices), given ", a->n);
  K k=kK(a)[2];
  if(k->t==KJ) {
   TORCH_CHECK(k->n==2, c,": expecting 2 vector indices, input & target, given ",k->n);
   x[0]=v->at(kJ(k)[0]); y[0]=v->at(kJ(k)[1]);
  } else {
   TORCH_CHECK(!k->t,   c,": expecting 2 sets of vector indices(inputs & targets), given ",kname(k));
   TORCH_CHECK(k->n==2, c,": expecting 2 sets of vector indices(inputs & targets), given ",k->n);
   for(J i=0; i<k->n; ++i)
    mvec(i,c,kK(k)[i],(i ? y : x),v);
  }
 }
}

// ---------------------------------------------------------------------------------
// mdict - index into dictionary of tensors as part of args to forward/backward, etc
// ---------------------------------------------------------------------------------
static Tensor* mdict(bool b,const char *c,TensorDict *d,S s) {
 // return tensor given symbol key, else signal not found
 Tensor *t=d->find(s);
 TORCH_CHECK(t, c,": unable to find ",(b ? "target" : "input")," key `",s);
 return t;
}

static void mdict(bool b,const char *c,K x,Tensors& t,TensorDict *d) {
 // attempt to set tensors in array t with key(s) into dictionary d from input x
 if(x->t == -KS) {
   t[0]=*mdict(b,c,d,x->s);
 } else if(x->t == KS) {
  TORCH_CHECK(x->n>0 && x->n<4, c,": expecting 1-3 ",(b ? "target" : "input")," keys, given ",x->n);
  for(J i=0; i<x->n; ++i)
   t[i]=*mdict(b,c,d,kS(x)[i]);
 } else {
  TORCH_ERROR(c,": ",(b ? "target" : "input")," keys expected, given ",kname(x));
 }
}

static void mdict(const char *c,K a,Tensors& x,Tensors& y,TensorDict *d) {
 if(a->n==2) {
  //TORCH_CHECK(v->size()==2, c,": unable to determine input & target given vector of ",v->size()," elements");
  //x[0]=v->at(0); y[0]=v->at(1);
 } else {
  TORCH_CHECK(a->n==3, c,": expecting 3 arguments, (model;vector;indices), given ", a->n);
  K k=kK(a)[2];
  if(k->t==KS) {
   TORCH_CHECK(k->n==2, c,": expecting 2 dictionary keys, input & target, given ",k->n);
   x[0]=*mdict(false,c,d,kS(k)[0]);
   y[0]=*mdict(true, c,d,kS(k)[1]);
  } else {
   TORCH_CHECK(!k->t,   c,": expecting 2 sets of dictionary keys(inputs & targets), given ",kname(k));
   TORCH_CHECK(k->n==2, c,": expecting 2 sets of dictionary keys(inputs & targets), given ",k->n);
   for(J i=0; i<k->n; ++i)
    mdict(i,c,kK(k)[i],(i ? y : x),d);
  }
 }
}

// ----------------------------------------------------------------
// tcount - return count of defined tensors in array
// margs - process k args into tensor array, vector or dictionary
// ----------------------------------------------------------------
static auto tcount(const Tensors& x) {
 size_t n=0; 
 for(const auto& a:x)
  if(a.defined())
   n++;
  else
   break;
 return n;
}

static void margs(const char *c,K x,Tensors& t,TensorVector *&v,TensorDict *&d) {
 size_t m=t.size(),n=tcount(t);
 TORCH_CHECK(n<m, c,": ",n," tensors already defined, additional arg(s) not implemented");
 if(auto *g=xtag(x)) {
  switch(g->a) {
   case Class::tensor: t[n]=((Kten*)g)->t; break;
   case Class::vector:
    TORCH_CHECK(!n, c,": tensor/vector mix not implemented");
    TORCH_CHECK(!v, c,": 2nd vector unexpected");
    v=&((Kvec*)g)->v;  break;
   case Class::dict:
    TORCH_CHECK(!n, c,": tensor/dictionary mix not implemented");
    TORCH_CHECK(!d, c,": 2nd dictionary unexpected");
    d=&((Kdict*)g)->d; break;
   default:
    TORCH_ERROR(c,": unexpected ",mapclass(g->a)," argument");
  }
 } else if(v) {
  IntArrayRef vi; size_t i=0;
  TORCH_CHECK(xsize(x,vi), c,": expecting vector indices, given ",kname(x));
  TORCH_CHECK(vi.size()>0 && vi.size()<=m, "forward: expecting 1-",m," vector indices, ",vi.size()," given");
  for(auto j:vi) t[i++]=v->at(j);
 } else if(d) {
  S s; J sn=xlen(x); Tensor *di;
  TORCH_CHECK(xsyms(x,s), c,": expecting dictionary key(s), given",(sn ? " " : " empty "),kname(x));
  TORCH_CHECK(sn>0 && sn<=m, c,": expecting 1-",m," dictionary keys, ",sn," given");
  for(J i=0;i<sn;++i) {
   if(i) s=kS(x)[i];
   TORCH_CHECK(di=d->find(s), c,": dictionary key `",s," not found");
   t[i]=*di;
  }
 } else if(xmixed(x,3)) {
   for(J i=0; i<x->n; ++i)
    margs(c,kK(x)[i],t,v,d);
 } else {
   t[n]=kput(x);
 }
}

static void margs(const char *c,K a,Tensors& x,Tensors& y,TensorVector *&v,TensorDict *&d) {
 margs(c,kK(a)[1],x,v,d);
 if(v) {
  mvec(c,a,x,y,v);
 } else if(d) {
 } else {
  margs(c,kK(a)[2],y,v,d);
 }
}

// ---------------------------------------------------------------------------
// addtensor - add tensor(s) to vector from given dictionary/vector
// modelarg - handle k inputs for forward calculations
// modelargs - handke k inputs for both forward & backward calculations
// ---------------------------------------------------------------------------
static void addtensor(TensorVector& a,const char *c,K x,TensorVector *v) {
 IntArrayRef i; TORCH_CHECK(xsize(x,i), c,": expecting vector indices, given ",kname(x));
 for(auto j:i) {
  TORCH_CHECK(-1<j && j<v->size(), c,": invalid vector index of ",j);
   a.emplace_back(v->at(j));
 }
}

static void addtensor(TensorVector& a,const char *c,K x,TensorDict *d) {
  S s; J n=xlen(x); Tensor *t;
  TORCH_CHECK(xsyms(x,s), c,": expecting dictionary key(s), given",(n ? " " : " empty "),kname(x));
  for(J i=0;i<n;++i) {
   if(i) s=kS(x)[i];
   TORCH_CHECK(t=d->find(s), c,": dictionary key `",s," not found");
   a.emplace_back(*t);
  }
}

static bool modelarg(TensorVector& a,size_t n,K x,const char *c,TensorVector *&v,TensorDict *&d) {
 bool b=true;Tensor *t;
 TORCH_CHECK(n<2, c,": argument(s) nested too deeply");
 if(v) {
  b=false; addtensor(a,c,x,v);
 } else if(d) {
  b=false; addtensor(a,c,x,d);
 } else if((t=xten(x)) || (v=xvec(x)) || (d=xtensordict(x))) {
  if(t) a.emplace_back(*t);
 } else if(xptr(x)) {
  TORCH_ERROR(c,": unexpected ",kname(x)," argument");
 } else if(xmixed(x,3)) {
  for(J i=0; i<x->n; ++i)
   TORCH_CHECK(modelarg(a,n+1,kK(x)[i],c,v,d) || i==x->n-1, c,": too many nested arg(s) given");
  b=false;
 } else {
   a.emplace_back(kput(x));
 }
 return b;
}

static TensorVector modelarg(K x,const char* c) {
 TensorVector a,*v=nullptr; TensorDict *d=nullptr; 
 for(J i=1; i<x->n; ++i) 
  TORCH_CHECK(modelarg(a,0,kK(x)[i],c,v,d) || i==x->n-1,
              c,": unexpected arg(s) i=",i," tensors defined: ",a.size());
 return a;
/*
 if(a.size()) return a;  // vector of inputs defined from vector/dict & indices/keys
 else if(v)   return v;  // vector given, no indices
 else if(d)   return d;  // dictionary given, no keys
 else TORCH_ERROR(c,": no tensors defined");
*/
}

static void modelargs(const char *c,K a,Tensors& x,Tensors& y,TensorVector *&v,TensorDict *&d) {
 margs(c,kK(a)[1],x,v,d);
 if(v) {
  mvec(c,a,x,y,v);
 } else if(d) {
 } else {
  margs(c,kK(a)[2],y,v,d);
 }
}

// ------------------------------------------------------------------------------
//  fwd calcs..
// ------------------------------------------------------------------------------
static Output fwd(Cast c,Module& m,const Tensors& x) {
 switch(tcount(x)) {
  case 1: return mforward(c,m,x[0]);
  case 2: return mforward(c,m,x[0],x[1]);
  case 3: return mforward(c,m,x[0],x[1],x[2]);
  default: TORCH_ERROR("no forward method implemented for ",tcount(x)," tensors");
 }
}

static Output fwd(Cast c,Module& m,const TensorVector& x) {
 switch(x.size()) {
  case 1: return mforward(c,m,x[0]);
  case 2: return mforward(c,m,x[0],x[1]);
  case 3: return mforward(c,m,x[0],x[1],x[2]);
  default: TORCH_ERROR("no forward method implemented for ",x.size()," tensors");
 }
}

KAPI margtest(K x) {
 KTRY
  auto *m=xmodel(x,0); auto *q=m ? nullptr : xmodule(x,0); Cast c=m ? m->mc : q->c;
  TORCH_CHECK((m || q) && x->n>1, "forward: requires a module or model and at least one input");
  return kout(fwd(c,m ? *m->m : *q->m, modelarg(x,"forward")));
 KCATCH("forward");
}

KAPI forward2(K a) {
 KTRY
  auto *m=xmodel(a,0); auto *q=m ? nullptr : xmodule(a,0); Cast c=m ? m->mc : q->c;
  TORCH_CHECK((m || q) && a->n>1, "forward: requires a module or model and at least one input");
  Tensors x; TensorVector *v=nullptr; TensorDict *d=nullptr;
  for(J i=1; i<a->n; ++i) margs("forward",kK(a)[i],x,v,d);
  if(v) {
   TORCH_CHECK(v->size(), "forward: empty vector given"); x[0]=v->at(0);
  } else if(d) {
   TORCH_ERROR("forward: unable to derive tensor inputs from dictionary without keys");
  }
  return kout(fwd(c,*(m ? m->m : q->m),x));
 KCATCH("forward");
}

KAPI back(K a) {
 KTRY
  auto *m=xmodel(a,0);
  TORCH_CHECK(m && a->n>2, "backward: expects model with inputs & targets");
  Tensors x,y; TensorVector *v=nullptr; TensorDict *d=nullptr;
  margs("backward",a,x,y,v,d);
  return (K)0;
 KCATCH("backward");
}

// -------------------------------------------------------------------------------------------
// add model api functions to library dictionary
// -------------------------------------------------------------------------------------------
void modelfn(K x) {
 fn(x, "model",      KFN(model),     1);
 fn(x, "zerograd",   KFN(zerograd),  1);
 fn(x, "forward",    KFN(forward),   1);
 fn(x, "backward",   KFN(kbackward), 1);
 fn(x, "train",      KFN(ktrain),    1);
 fn(x, "training",   KFN(training),  1);
 fn(x, "evaluate",   KFN(evaluate),  1);
 fn(x, "clip",       KFN(clip),      1);
 fn(x, "clipv",      KFN(clipv),     1);
}

