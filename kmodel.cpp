#include "ktorch.h"

// -------------------------------------------------------------------------------------------
// modelpart - parse args from k to define module, loss & optimizer
// modelkeys - return list of symbols used for model state dictionary
// modelstate - return a dictionary with state of module, loss fn & optimizer
// model - create model from module, loss & optimizer or retrieve input options
// -------------------------------------------------------------------------------------------
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
    if(q) m->mc=q->c, m->r=q->r, m->m=q->m;  //assign new module
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
// mforward - return tensor from running forward calcs on input(s)
// forward - forward calcs on sequential module or model
// -------------------------------------------------------------------------------------------
static Tensor mforward(Kmodel *m,const Tensor& x) {return mforward(m->mc,*m->m,x);}
static Tensor mforward(Kmodel *m,const TensorVector& v) {return mforward(m->mc,*m->m,v[0]);}

static size_t tcount(const Tensor& x,const Tensor& y,const Tensor& z) {
 if(x.defined() && !y.defined() && !z.defined())
  return 1;
 else if(x.defined() && y.defined() && !z.defined())
  return 2;
 else if(x.defined() && y.defined() && z.defined())
  return 3;
 else if(!x.defined())
  TORCH_ERROR("forward: unrecognized tensor arg(s), expecting x, (x;y) or (x;y;z), but initial x tensor not defined");
 else
  TORCH_ERROR("forward: unrecognized tensor arg(s), expecting x, (x;y) or (x;y;z), but given (x;z)");
}

K mforward(Cast c,Result r,Module& m,const Tensor& x,const Tensor& y={},const Tensor& z={});
K mforward(Cast c,Result r,Module& m,const Tensor& x,const Tensor& y,   const Tensor& z) {
 switch(r) {
  case Result::tensor:
   switch(tcount(x,y,z)) {
    case 1: return kten(mforward(c,m,x));
    case 2: return kten(mforward(c,m,x,y));
    case 3: return kten(mforward(c,m,x,y,z));
    default: TORCH_ERROR("forward: unrecognized tensor arg(s)");
   }
  case Result::vector:
  case Result::tuple:
  case Result::nested:
   return kvec(vforward(c,r,m,x,y,z));
  case Result::none:
   TORCH_ERROR("forward: no forward calculation defined for ",msym(c)," module");
  case Result::undefined:
   TORCH_ERROR("forward: no result type defined for ",msym(c)," module's forward calculation");
  default: TORCH_ERROR("forward: unrecognized result enumeration(",(I)r,") for ",mlabel(m));
 }
}

static K mforward(Cast c,Result r,Module& m,K a) {
 Tensor x,y,z; TensorVector *v;
 if((v=xvec(a,1))) {
  TORCH_CHECK(v->size(), "forward: empty vector of tensors supplied");
  IntArrayRef i;
  if(a->n==2) {
   return mforward(c, r, m, v->at(0));
  } else if(a->n==3 && xsize(a,2,i)) {
   switch(i.size()) {
    case 1: return mforward(c, r, m, v->at(i[0]));
    case 2: return mforward(c, r, m, v->at(i[0]), v->at(i[1]));
    case 3: return mforward(c, r, m, v->at(i[0]), v->at(i[1]), v->at(i[2]));
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
  return a->n==2 ? mforward(c,r,m,x) : (a->n==3 ? mforward(c,r,m,x,y) : mforward(c,r,m,x,y,z));
 }
}

KAPI forward(K x) {
 KTRY
  Ktag *g=xtag(x,0);
  TORCH_CHECK(g, "forward: expects module/model as first arg, with tensor(s)/vector/dictionary as additional args");
  switch(g->a) {
   case Class::module: {auto *m=(Kmodule*)g; return mforward(m->c, m->r,*m->m,x);}
   case Class::model:  {auto *m=(Kmodel*)g;  return mforward(m->mc,m->r,*m->m,x);}
   default: TORCH_ERROR("forward not implemented for ",mapclass(g->a));
  }
 KCATCH("forward");
}

// -------------------------------------------------------------------------------------------
// mbackward - given model, input & target, do forward calcs, get loss, backward prop on loss
// mloss - given model and vector of inputs, e.g. v=x,y, loss=loss(module(v[0]),v[1])
// -------------------------------------------------------------------------------------------
K mbackward(K a) {
 Kmodel *m=xmodel(a,0); Tensor *x,*y,r; TensorVector *v;
 TORCH_CHECK(m, "backward: first argument not a model");
 if((x=xten(a,1)) && (y=xten(a,2)) && a->n==3) {
  r=lossfwd(m->lc,*m->l,mforward(m->mc,*m->m,*x),*y);
 } else if ((v=xvec(a,1)) && a->n==2) {
  TORCH_CHECK(v->size()>1, "backward: vector expected to contain two or more tensors, given ",v->size());
  r=lossfwd(m->lc,*m->l,mforward(m->mc,*m->m,v->at(0)),v->at(1));
 } else {
  TORCH_ERROR("backward expects (model; inputs; targets) or (model;vector)");
 }
 r.backward();
 return kget(r);
}

Tensor mloss(Kmodel *m,const Tensor& x,const TensorVector &v) {
 if(v.size()==2)
  return lossfwd(m->lc,*m->l,x,v[1]);
 else if(v.size()==3)
  return lossfwd(m->lc,*m->l,x,v[1],v[2]);
 else
  TORCH_ERROR("model: ", v.size()," inputs given, expecting 2-3");
}

Tensor mloss(Kmodel *m,const TensorVector &v) {return mloss(m,mforward(m,v),v);}
Tensor mloss(Kmodel *m,const Tensor& x,const Tensor& y) {return lossfwd(m->lc,*m->l,mforward(m,x),y);}

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
static Tensor evalfwd(Cast c,Module& m,Tensor& x,int64_t w) {
 bool b=m.is_training(); Tensor y;
 if(b) m.train(false);               // turn off training mode
 if(w) {                             // if batches of window size w
  auto n=maxsize(x);                 // get maxsize
  TensorVector r;
  for(int64_t i=0; i<n; i+=w) {      // batches of w
   subset(x,0,i,w,n);
   r.emplace_back(mforward(c,m,x));  // accumulate forward calcs
  }
  subset(x,0,0,n,n);                 // restore size of inputs
  y=torch::cat(r);                   // and join batch results
 } else {
  y=mforward(c,m,x);                 // no batching, run forward on full inputs
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
 Tensor *t=d->find(s);
 TORCH_CHECK(t, c,": unable to find ",(b ? "target" : "input")," key `",s);
 return t;
}

static void mdict(bool b,const char *c,K x,Tensors& t,TensorDict *d) {
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

// -------------------------------------------------------------------------------------------
// tcount - return count of defined tensors in array
// -------------------------------------------------------------------------------------------
static auto tcount(const Tensors& x) {
 size_t n=0; 
 for(const auto& a:x)
  if(a.defined())
   n++;
  else
   break;
 return n;
}

static Output fwd(Cast c,Result r,Module& m,const Tensors& x) {
 switch(tcount(x)) {
  case 1: return mForward(c,m,x[0]);
  case 2: return mForward(c,m,x[0],x[1]);
  case 3: return mForward(c,m,x[0],x[1],x[2]);
  default: TORCH_ERROR("no forward method implemented for ",tcount(x)," tensors");
 }
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

