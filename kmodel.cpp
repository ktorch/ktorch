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

// -------------------------------------------------------------------------------------------
// mforward - return tensor from running forward calcs on input(s)
// mbackward - given model, input & target, do forward calcs, get loss, backward prop on loss
// mloss - given model and vector of inputs, e.g. v=x,y, loss=loss(module(v[0]),v[1])
// -------------------------------------------------------------------------------------------
Tensor mforward(Kmodel *m,const Tensor& x) {return mforward(m->mc,*m->m,x);}
Tensor mforward(Kmodel *m,const TensorVector& v) {return mforward(m->mc,*m->m,v[0]);}

K mbackward(K a) {
 Kmodel *m; Tensor *x,*y,r;
 if((m=xmodel(a,0)) && (x=xten(a,1)) && (y=xten(a,2))) {
  r=lossfwd(m->lc,*m->l,mforward(m->mc,*m->m,*x),*y);
 } else {
  TORCH_ERROR("backward expects (model; inputs; targets)");
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

KAPI ganstep(K a) {
 KTRY
  Kmodel *d=xmodel(a,0), *g=xmodel(a,1);
  TORCH_CHECK(d && g, "ganstep: supply discriminator & generator model as 1st & 2nd args");
  Tensor* x=xten(a,1); Tensor* y=xten(a,2); Tensor* z=xten(a,3);
  d->o->zero_grad();
  Tensor l0=mloss(d, *x, (*y)[0]);
  l0.backward();
  Tensor gx=mforward(g->mc,*g->m,*z);
  Tensor l1=mloss(d, gx.detach(), (*y)[1]);
  l1.backward();
  optstep(d);
  g->o->zero_grad();
  Tensor l2=mloss(d, gx, (*y)[2]);
  l2.backward();
  optstep(g);
  return(K)0;
 KCATCH("ganstep");
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
 for(auto& m:env().metric) 
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

// -------------------------------------------------------------------------------------------
// add model api functions to library dictionary
// -------------------------------------------------------------------------------------------
void modelfn(K x) {
 fn(x, "model",      KFN(model),     1);
 fn(x, "train",      KFN(ktrain),    1);
 fn(x, "training",   KFN(training),  1);
 fn(x, "evaluate",   KFN(evaluate),  1);
}

/*
 main functions:

 step
 train/evaluate
 forward
 backward/loss

 train(m; v; ix; iy; window; epochs; shuffle)   / train(m; v; 0; 1; 30; 1; 1b)
 train(m; d; kx; ky; window; epochs; shuffle)   / train(m; d;`x;`y; 30; 1; 1b)

 train(m; d; (kx;ky);   train(m;d;(`x1`x2;`y);30)
 train(m;(d;`x); (d;`y); 
 train(m; (x1;x2); y; ..)

 forward(m; x)
 forward(m; z; y)
 forward(m; x; y; z)
 forward(m; v)
 forward(m; v; i)
 

 forward(m; (v;0 1); t)

 backward(m; 

-------------------------------------------------------------------

 auto step=[&](Optimizer& o, Sequential m, Tensor x, Tensor y) {
     auto f=[&]() {
       o.zero_grad();
       auto x=m->forward(x);
       auto l=torch::binary_cross_entropy(x,y);
       l.backward();
       return l;
     };
     return o.step(f);
   };
 Tensor loss=step(o,m,x,y);

*/

