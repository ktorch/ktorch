#include "ktorch.h"
#include <torch/csrc/autograd/function.h>

// ---------------------------------------------------------------------------
// kten - given tensor ref, return ptr to struct w'attrs, void ptr to tensor
// kvec - given reference to vector of tensors, return ptr to struct w'attrs
// kdict - given tensor dictionary reference, return ptr to containing struct
// ---------------------------------------------------------------------------
K kten(const Tensor& t) {return kptr(new Kten(t));}
K kvec(const TensorVector& v) {return kptr(new Kvec(v));}
K kdict(const TensorDict &d) {return kptr(new Kdict(d));}

// -------------------------------------------------------------------------
// razeflag - check if general list made up entirely of scalars
// razelist - if general list is all scalars, raze to simple list
// -------------------------------------------------------------------------
static bool razeflag(K x) {
 if(!x->t && x->n>0) {
  auto t=kK(x)[0]->t;
  if(t>=0) 
   return false;
  for(J i=1; i<x->n; ++i)
   if(kK(x)[i]->t != t)
    return false;;
  return true;
 } else {
  return false;
 }
}

static K razelist(K x) {
 if(razeflag(x)) {
  J i; K y=ktn(-kK(x)[0]->t, x->n);
  switch(y->t) {
   case KE: for(i=0; i<y->n; ++i) kE(y)[i]=kK(x)[i]->e; break;
   case KF: for(i=0; i<y->n; ++i) kF(y)[i]=kK(x)[i]->f; break;
   case KJ: for(i=0; i<y->n; ++i) kJ(y)[i]=kK(x)[i]->j; break;
   case KI: for(i=0; i<y->n; ++i) kI(y)[i]=kK(x)[i]->i; break;
   case KH: for(i=0; i<y->n; ++i) kH(y)[i]=kK(x)[i]->h; break;
   case KB:
   case KC:
   case KG: for(i=0; i<y->n; ++i) kG(y)[i]=kK(x)[i]->g; break;
   default: TORCH_ERROR("unable to raze general list -> ",kname(y));
  }
  return r0(x), y;
 } else {
  return x;
 }
}

// -----------------------------------------------------------------------------------------
// cpermute - permute real representation of complex tensor from (real,'imag) -> (real;imag)
// sparsereal - convert sparse complex tensor to sparse real representation (real,'imag)
// toreal - convert complex tensor to real representation, 1 complex number to real & imag
// -----------------------------------------------------------------------------------------
static Tensor cpermute(const Tensor& x) {
 std::vector<int64_t> d;
 for(int64_t i=0; i<x.dim(); ++i) d.push_back(i-1);
 return x.permute(d);
}

static Tensor sparsereal(const Tensor& t) {
 // asof version 1.8.1, cannot go from sparse complex -> dense complex
 // so attempt sparse complex -> sparse real representation -> dense real representation
 auto n=t.sizes().vec(); n.push_back(2);
 return torch::sparse_coo_tensor(t._indices(),torch::view_as_real(t._values()),n);
}

static Tensor toreal(const Tensor& t, c10::optional<bool> b=c10::nullopt);
static Tensor toreal(const Tensor& t, c10::optional<bool> b) {
 bool c=b ? *b : env().complexfirst;
 return c ? cpermute(t.is_sparse() ? sparsereal(t).to_dense() : torch::view_as_real(t))
          :          t.is_sparse() ? sparsereal(t).to_dense() : torch::view_as_real(t);
}

// -------------------------------------------------------------------------
// kgetscalar - return k scalar given a scalar tensor
// kgets - process tensor at depth, creating k array
// kget - take tensor reference, return k scalar/array
//      - take reference to vector of longs/doubles, return k list
//      - take reference to vector/deque of tensors, return k lists
//      - take reference to dictionary of tensors, return k dictionary
// -------------------------------------------------------------------------
static K kgetscalar(const Tensor &t){
 auto s=t.item();
 switch(t.scalar_type()) {
  case torch::kFloat:  return ke(s.toFloat());
  case torch::kDouble: return kf(s.toDouble());
  case torch::kHalf:   return ke(s.toFloat());
  case torch::kShort:  return kh(s.toShort());
  case torch::kInt:    return ki(s.toInt());
  case torch::kLong:   return kj(s.toLong());
  case torch::kBool:   return kb(s.toBool());
  case torch::kByte:   return kg(s.toByte());
  case torch::kChar:   return kc(s.toChar());
  default: TORCH_ERROR("unrecognized scalar tensor type: ", t.dtype(), ", cannot return k scalar"); return (K)0;
 }
}

static K kgets(I i,I j,Ktype k,J b,const int64_t *s,S &p) {
//i:depth, j:max depth, k:k type, b:bytes to copy, s:sizes, p:data ptr
 K x=ktn((i<j) ? 0 : k,s[i]);                     //create k list
 if(x->t) {                                       //if base type
  if(x->n) {                                      //   and non-zero length
   memcpy(kG(x),p,b);                             //copy k <- tensor
   p+=b;                                          // and incr data ptr
  }
 } else {                                         // else
   for(J y=0;y<x->n;++y)                          // call down a level
    kK(x)[y]=kgets(i+1,j,k,b,s,p);                // until base data type
 }
 return x;
}

K kget(const Tensor &t) {
 if(!t.defined())
  return (K)0;   // NULL TENSOR ktn(0,0);
 else if (t.is_complex())
  return kget(toreal(t));
 else if (!t.dim())      // if 0-dimensional
  return kgetscalar(t);  // return scalar
 Tensor c;
 if(t.dtype()==torch::kHalf)
  c=t.toType(torch::kFloat).contiguous().toBackend(torch::Backend::CPU);
 else if (t.layout()==torch::kSparse)
  c=t.to_dense().toBackend(torch::Backend::CPU);
 else
  c=t.contiguous().toBackend(torch::Backend::CPU);
 I j=c.dim()-1; const int64_t *s=c.sizes().data();  // dimension & sizes at each dim
 J b=s[j]*c.element_size();                   // bytes to copy at lowest depth
 S p=(S)c.data_ptr();                         // contiguous data pointer
 return kgets(0,j,maptype(t.dtype()),b,s,p);
}

K kget(const LongVector& v)   {return klist(v.size(),v.data());}
K kget(const DoubleVector& v) {return klist(v.size(),v.data());}

K kget(const TensorDeque& v) {
 K x=ktn(0,v.size());
 for(size_t i=0; i<v.size(); ++i) kK(x)[i]=kget(v[i]);
 return x;
}

K kget(const TensorVector& v,K x) { // x-nullptr by default, else indices
 if(!x) {
  K r=ktn(0,v.size());
  for(J i=0; i<r->n; ++i) kK(r)[i]=kget(v[i]);
  return razelist(r);
 } else if(x->t == -KJ) {
  return kget(v.at(x->j));
 } else if(x->t == KJ) {
  K r=ktn(0,x->n);
  for(J i=0; i<x->n; ++i) kK(r)[i]=kget(v.at(kJ(x)[i]));
  return razelist(r);
 } else {
  TORCH_ERROR("vector: expecting 2nd arg of long indices, given ",kname(x));
 }
}

K kget(const TensorDict& d,K x) { // x-nullptr by default, can contain sym(s) for indexing
 if(!x) {
  J i=0; K k=ktn(KS,d.size()),v=ktn(0,d.size());
  for(const auto &a:d) {
   kS(k)[i]=cs(a.key().c_str());
   kK(v)[i]=kget(a.value());
   ++i;
  }
  return xD(k,razelist(v));
 } else if(x->t == -KS) {
  return kget(d[x->s]);
 } else if(x->t == KS) {
  K r=ktn(0,x->n);
  for(J i=0; i<x->n; ++i)
   kK(r)[i]=kget(d[kS(x)[i]]);
  return razelist(r);
 } else {
  TORCH_ERROR("dict: expecting 2nd arg of symbol(s) for indexing, given ",kname(x));
 }
}

// ------------------------------------------------------------------------
// kget - return module output as array or list of arrays of tensor values
// tvec - return vector of tensors from tensor array
// kout - return module output as tensor or vector of tensors
// ------------------------------------------------------------------------
K kget(const Tuple& t) {
 return knk(2,kget(std::get<0>(t)),kget(std::get<1>(t)));
}

K kget(const Tensors& t) {
 J i=0; K x=ktn(0,std::tuple_size<Tensors>::value);
 for(const auto& a:t) kK(x)[i++]=kget(a);
 return x;
}

K kget(const Output& o) {
 if(auto a=c10::get_if<Tensor>(&o)) {
  return kget(*a);
 } else if(auto a=c10::get_if<Tuple>(&o)) {
  return kget(*a);
 } else if(auto a=c10::get_if<Tensors>(&o)) {
  return kget(*a);
 } else if(auto a=c10::get_if<TensorVector>(&o)) {
  return kget(*a);
 } else {
  TORCH_ERROR("unrecognized output from forward calculation");
 }
}

TensorVector tvec(const Tensors& t) {
 return {t[0],t[1],t[2]};
}

K kout(const Output& o) {
 if(auto a=c10::get_if<Tensor>(&o)) {
  return kten(*a);
 } else if(auto a=c10::get_if<Tuple>(&o)) {
  return kvec({std::get<0>(*a),std::get<1>(*a)});
 } else if(auto a=c10::get_if<Tensors>(&o)) {
  return kvec(tvec(*a));
 } else if(auto a=c10::get_if<TensorVector>(&o)) {
  return kvec(*a);
 } else {
  TORCH_ERROR("unrecognized output from forward calculation");
 }
}

// -------------------------------------------------------------------------------
// checkint - check options for modes not implemented for integral types
// checksparse - check options for sparse tensor, signal nyi combinations
// to - change tensor/vector device/type, create new tensor if copy flag set
// ktenpair - given a pair of tensors return pair of pointers or array
// kten3 - given a triplet of tensors return triplet of pointers or array
// -------------------------------------------------------------------------------
static bool checkint(const TensorOptions& o,Tensormode m=Tensormode::undefined);
static bool checkint(const TensorOptions& o,Tensormode m) {
 if(o.has_dtype() && torch::isIntegralType(torch::typeMetaToScalarType(o.dtype()),true)) {
  switch(m) {
   case Tensormode::rand:
   case Tensormode::randn:
    TORCH_ERROR(modesym(m), ": not implemented for ",optdtype(o.dtype())," tensors");
   default: break;
  }
  return true;
 } else {
  return false;
 }
}

static bool checksparse(const TensorOptions& o,Tensormode m=Tensormode::undefined);
static bool checksparse(const TensorOptions& o,Tensormode m) {
 if(o.layout()==torch::kSparse || m==Tensormode::sparse) {
  TORCH_CHECK(!o.has_layout() || o.layout()==torch::kSparse, "tensor: sparse mode incompatible with layout set to ",optlayout(o.layout()));
  TORCH_CHECK(!o.pinned_memory(), "sparse tensors cannot have pinned memory");
  TORCH_CHECK(!(o.has_memory_format() && (o.memory_format_opt().value()==torch::MemoryFormat::ChannelsLast ||
                                          o.memory_format_opt().value()==torch::MemoryFormat::ChannelsLast3d)),
              "sparse tensors cannot use memory formats with channels as last dimension");
  switch(m) {
   case Tensormode::undefined:
   case Tensormode::complex:
   case Tensormode::empty:
   case Tensormode::sparse:
   case Tensormode::zeros:
    break;
   default: TORCH_ERROR(modesym(m), ": not implemented for sparse tensors");
  }
  return true;
 } else {
  return false;
 }
}

static Tensor to(const Tensor& t,const TensorOptions& o,bool a=false,bool b=false);
static Tensor to(const Tensor& t,const TensorOptions& o,bool a,bool b) {
 // as of version 1.8.0, errors using to() with sparse or gradients:
 // pinned memory doesn't seem to be handled from within any .to() method
 //"Operators taking TensorOptions cannot take a TensorOptions with options.requires_grad set as true. This isn't implemented yet."
 //"to(options) doesn't support converting to a different layout, but got self.layout being Strided and options.layout set as Sparse"
 if(!t.defined())
  return t;
 else if(checksparse(o))
   return t.to_sparse().to(o.requires_grad(false),a,b).set_requires_grad(o.requires_grad());
 else if(t.is_sparse() && o.has_layout() && o.layout()==torch::kStrided)
   return to(t.to_dense(),o);
 else if(o.pinned_memory())
  return t.to(o.requires_grad(false),a,b).pin_memory().set_requires_grad(o.requires_grad());
 else
  return t.to(o.requires_grad(false),a,b).set_requires_grad(o.requires_grad());
}

K to(Kten* t,const TensorOptions& o,bool a,bool b) {
 TORCH_CHECK(t->t.defined(),"to: cannot change attribute(s) of an undefined tensor");
 auto r=to(t->t,o,a,b);
 if(b)                 // if copy flag set
  return kten(r);      // return new tensor
 if(!t->t.is_same(r))  // else if device/dtype caused new tensor
  t->t=r;              // replace tensor in k ptr
 return (K)0;
}

void to(TensorDict& d,const TensorOptions& o,bool a) {
 for(auto& i:d) {
  if(i.value().defined()) {
   auto t=i.value().to(o,a);
   if(!i.value().is_same(t)) i.value()=std::move(t);
  }
 }
}

void to(TensorVector& v,const TensorOptions& o,bool a) {
 for(auto& t:v) {
  if(t.defined()) {
   auto r=t.to(o,a);
   if(!t.is_same(r)) t=std::move(r);
  }
 }
}

K to(Kdict* d,const TensorOptions& o,bool a) {to(d->d,o,a); return (K)0;}
K to(Kvec*  v,const TensorOptions& o,bool a) {to(v->v,o,a); return (K)0;}

K ktenpair(bool p,Tensor& a,Tensor& b) {  // p:true if returning tensor pointers
 if(p) return knk(2,kten(a),kten(b));
 else  return knk(2,kget(a),kget(b));
}

K kten3(bool p,Tensor& a,Tensor& b,Tensor& c) {  // p:true if returning tensor pointers
 if(p) return knk(3,kten(a),kten(b),kten(c));
 else  return knk(3,kget(a),kget(b),kget(c));
}

// ---------------------------------------------------------------------------------------
// kputscalar - copy single k value to CPU tensor scalar
// kdepth - check k array at depth for consistent datatype, size, etc, throw errors
// kputs - descend depth of k array, determining dim & sizes, copying data types to tensor
// kput - controlling function to read k array, create tensor and copy data at depth
// ---------------------------------------------------------------------------------------
void kputscalar(K x,Tensor &t) {
 Scalar s;
 TORCH_CHECK(xscalar(x,s), "unable to translate k ",kname(x->t)," to scalar tensor");
 t=torch::full({},s,maptype(x->t));
}

static void kdepth(K x,I i,H k,Ksize &s){
 if(x->t < 0) {
  TORCH_ERROR("unable to map mixed array to tensor: ",kname(x->t)," encountered at depth ",i);
 } else if(k != nh) {             // if base type already encountered
  I j=s.size()-1;                 // last size index
  if(x->n != s[i]) {              // check that dimensions are consistent
   TORCH_ERROR("dimension mismatch at depth ",i,", ",s[i]," vs ",x->n);
  } else if(x->t != (i<j ? 0 : k)) {  // check for same data type at same depth
   TORCH_ERROR("type mismatch at depth ",i,", ",kname(i<j ? 0 : k)," vs ",kname(x->t));
  }
 } else {
  s.push_back(x->n);              // no error, no base type yet, accumulate sizes
 }
}

static void kputs(K x,I i,H &k,Ksize &s,J &b,S &p,Tensor &t) {
 kdepth(x,i,k,s);
 if(x->t || !x->n) {     // if base data type or empty
  if(k==nh)  {           // if first encounter w'base data type
   k=x->t;
   t=k ? torch::empty(s, maptype(k)) : torch::empty(s);
   b=t.element_size() * s[i];  // bytes to copy
   p=(S)t.data_ptr();          // contiguous data pointer
  }
  memcpy(p,kG(x),b); p+=b;
 } else {
  for(I j=0;j<x->n;++j) kputs(kK(x)[j],i+1,k,s,b,p,t);
 }
}

Tensor kput(K x) {        
 H k=nh;                   // fill w'base data type for nested k value
 J b=0;                    // fill w'bytes to copy
 Ksize s;                  // fill w'k array size at each depth
 S p=nullptr;              // data pointer for created tensor
 Tensor t;                 // undefined tensor
 if(x->t < 0)              // if scalar
  kputscalar(x,t);         // create scalar backed by tensor
else if(!xnull(x))       // else go through the depth of the array
// else                      // else go through the depth of the array NULL TENSOR
  kputs(x,0,k,s,b,p,t);    // until base data type encountered
 return t;
}

Tensor kput(K x,J i) {
 if(xind(x,i)) 
  return kput(kK(x)[i]);
 else
  TORCH_ERROR("unable to index ",kname(x->t),", element: ",i);
}

// --------------------------------------------------------------------
// kput - given indices & values from k, [re]set TensorVector elements
// --------------------------------------------------------------------
static void kput(TensorVector& v,J i,const Tensor& t) {
 if(i == nj || i == -1)
  v.emplace_back(t);
 else
  v.at(i)=t;
}

static bool kput(TensorVector& v,J i,K x) {
 Tensor *t=nullptr;
 if(xptr(x)) {
  t=xten(x);
  TORCH_CHECK(t, "vector: not implemented for ",kname(x));
 }
 kput(v, i, t ? *t : kput(x));
 return t;
}

static void kput(TensorVector& v,K x,K y) {
 if(x->t == -KJ) {
  if(kput(v,x->j,y))
   kfree(y);
 } else if(x->t == KJ) {
  if(y->t) {
   Tensor t=kput(y);
   TORCH_CHECK(x->n == t.numel(), "vector: length error, index count of ", x->n, " with ", t.numel(), " value(s)");
   for(J i=0; i<x->n; ++i)
    kput(v,kJ(x)[i],t.dim() ? t[i].clone() : t);
  } else {
   TORCH_CHECK(x->n == y->n, "vector: length error, index count of ", x->n, " with ", y->n, " value(s)");
   bool b=false; TensorVector w;
   for(J i=0; i<x->n; ++i) if(kput(w, -1, kK(y)[i])) b=true; // put k values/tensor ptrs in temp vector
   for(J i=0; i<x->n; ++i) kput(v,kJ(x)[i],w[i]);            // if no error, add to existing vector
   if(b)                                                     // if any tensor pointers, free
    for(J i=0; i<y->n; ++i)
     if(xten(y,i)) kfree(y,i);
  }
 } else {
  TORCH_ERROR("vector: expecting long indices as 2nd arg, given ",kname(x));
 }
}

// ------------------------------------------------------------------------
// kput - put tensors/array in dictionary using symbols and arrays/tensors
// ------------------------------------------------------------------------
static void kput(TensorDict& d,S s,const Tensor& t) {
 if(d.contains(s))
  d[s]=std::move(t);
 else
  d.insert(s,std::move(t));
}

static bool kput(TensorDict& d,S s,K x) {
 Tensor* t=nullptr;
 if(xptr(x)) {
  t=xten(x);
  TORCH_CHECK(t, "dict: not implemented for ",kname(x));
 }
 kput(d,s,t ? *t : kput(x));
 return t;
}

static void kput(TensorDict& d,K x,K y) {
 if(x->t == -KS) {
  if(kput(d,x->s,y))
   kfree(y);
 } else if(x->t == KS) {
  if(y->t) {
   Tensor t=kput(y);
   TORCH_CHECK(x->n == t.numel(), "dict: length error, ", x->n, " key(s) with ", t.numel(), " value(s)");
   for(J i=0; i<x->n; ++i)
    kput(d,kS(x)[i],t.dim() ? t[i].clone() : t);
  } else {
   TORCH_CHECK(x->n == y->n, "dict: length error, ", x->n, " key(s) with ", y->n, " value(s)");
   bool b=false; TensorVector w;
   for(J i=0; i<x->n; ++i) if(kput(w, -1, kK(y)[i])) b=true;  // add tensors/arrays to temp vector
   for(J i=0; i<x->n; ++i) kput(d, kS(x)[i], w[i]);           // if no error, add to dictionary
   if(b)
    for(J i=0; i<y->n; ++i)
     if(xten(y,i)) kfree(y,i);
  }
 } else {
  TORCH_ERROR("dict: given ptr, expecting symbol keys & values, but 2nd arg is ",kname(x));
 }
}

TensorDict kputd(K x) {
 TensorDict d;
 if(xdict(x) || (x->n==2 && (kK(x)[0]->t==KS || kK(x)[0]->t==-KS)))
  kput(d,kK(x)[0],kK(x)[1]);
 else if(!xempty(x))
  TORCH_ERROR("dict: expecting k dictionary or (syms;vals), given ",kname(x));
 return d;
}
 
// --------------------------------------------------------------------------------------
// complextype - get component data type from complex data type or default data type
// complexdim  - determine first/last dim on which to split for real/imaginary
// --------------------------------------------------------------------------------------
static ScalarType complextype(c10::optional<TypeMeta> t,ScalarType a=ScalarType::Undefined, ScalarType b=ScalarType::Undefined);
static ScalarType complextype(c10::optional<TypeMeta> t,ScalarType a,ScalarType b) {
 ScalarType d;
 if(torch::isFloatingType(a) && torch::isFloatingType(b)) {
  TORCH_CHECK(a==b, "complex: real input is ",optdtype(a),", imaginary input is ",optdtype(b));
 }
 if(t) {
  switch(torch::typeMetaToScalarType(*t)) {
   case torch::kComplexHalf:   d=torch::kHalf; break;
   case torch::kComplexFloat:  d=torch::kFloat; break;
   case torch::kComplexDouble: d=torch::kDouble; break;
   default:
    TORCH_ERROR("unable to create complex tensor with given datatype: ",optdtype(*t));
  }
 } else if(torch::isFloatingType(a)) {
  d=a; 
 } else if(torch::isFloatingType(b)) {
  d=b; 
 } else {
  d=torch::get_default_dtype_as_scalartype();
  if(!isFloatingType(d)) d=torch::kFloat;
 }
 return d;
}

static int64_t complexdim(const Tensor& a,c10::optional<bool> b) {
 bool c=b ? *b : env().complexfirst; int64_t d=c ? 0 : -1;
 TORCH_CHECK(a.dim(),      "complex: single input array must have one or more dimensions");
 TORCH_CHECK(a.size(d)==2, "complex: single input array must have a ",
                           (c ? "first" : "last")," dimension of size 2 (real",(c ?  ";" : ",'"),
                           "imaginary), given size of ",a.sizes());
 return d;
}

// --------------------------------------------------------------------------------------
// complex1 - make a complex tensor from single input: (real,'imag) or (real;imag)
// --------------------------------------------------------------------------------------
static Tensor complex1(const Tensor& a,int64_t d,Tensor& r) {  // w'output tensor
 return torch::complex_out(r, a.select(d,0), a.select(d,1));
}

static Tensor complex1(const Tensor& a,Tensor& r,c10::optional<bool> b=c10::nullopt);
static Tensor complex1(const Tensor& a,Tensor& r,c10::optional<bool> b) {
 return complex1(a.to(complextype(r.dtype())), complexdim(a,b), r);
}

static Tensor complex1(const Tensor& a,int64_t d) {   // no output tensor
 return torch::complex(a.select(d,0), a.select(d,1));
}

static Tensor complex1(const Tensor& a,const TensorOptions& o,c10::optional<bool> b=c10::nullopt);
static Tensor complex1(const Tensor& a,const TensorOptions& o,c10::optional<bool> b) {
 return complex1(a.to(complextype(o.dtype_opt(),a.scalar_type())), complexdim(a,b)).to(o);
}

// --------------------------------------------------------------------------------------
// complex2 - make a complex tensor from real & imaginary inputs
// --------------------------------------------------------------------------------------
static Tensor complex2(const Tensor& a,const Tensor& b,Tensor& r) {  // w'output tensor
 auto t=complextype(r.dtype(), a.scalar_type(), b.scalar_type());
 return torch::complex_out(r,a.to(t),b.to(t));
}

static Tensor complex2(const Tensor& a,const Tensor& b,TensorOptions& o) {
 auto t=complextype(o.dtype_opt(), a.scalar_type(), b.scalar_type());
 return torch::complex(a.to(t), b.to(t));
}


// ----------------------------------------------------------------------------------------
// tensorlike - tensor creation routines, e.g. ones_like() where tensor given as template
// tensorout - tensor creation routines, e.g. ones_out(), where output tensor is given
// tensoropt - tensor creation routines where tensor size and option(s) given
// tensormode - determines whether a template tensor or output tensor given w'other args
// tensorput - put k value(s) -> tensor, return new tensor ptr unless output tensor given
// tensorget - given tensor ptr and optional flag & indexing args, get result for k array
// vectorptr - given vector ptr, return tensor pointers, or single pointer if index given
// dictptr - given dictionary ptr, return dictionary of tensor pointers, or single pointer
// tensor - high level function to create/retrieve/move/recast tensor from k
// ----------------------------------------------------------------------------------------
static void tensorlike(K x,Tensormode m,const Tensor &t,Tensor &r) {  // t:input, r:result tensor
 J i,j; Scalar s; TensorOptions o;
 bool b=xopt(x,x->n-1,o); I nx=x->n-b;  //set flag if options given, count non-option args
 switch(m) {
  case Tensormode::empty: if(nx==2) r=torch::empty_like(t,o); break;
  case Tensormode::zeros: if(nx==2) r=torch::zeros_like(t,o); break;
  case Tensormode::ones:  if(nx==2) r=torch::ones_like(t,o);  break;
  case Tensormode::rand:  if(nx==2) r=torch::rand_like(t,o);  break;
  case Tensormode::randn: if(nx==2) r=torch::randn_like(t,o); break;
  case Tensormode::full:
   if(nx==3 && xscalar(x,2,s))
    r=torch::full_like(t,s,o.has_dtype() ? o : o.dtype(maptype(kK(x)[2]->t)));
   break;
  case Tensormode::randint:
   if     (nx==3 && xlong(x,2,j))                 r=torch::randint_like(t,j,o);
   else if(nx==4 && xlong(x,2,i) && xlong(x,3,j)) r=torch::randint_like(t,i,j,o);
   break;
  default:
   TORCH_ERROR("tensor: mode `",kK(x)[0]->s," not implemented with input tensors");
   break;
 }
}

static void tensorout(K x,Tensormode m,Tensor &t,Tensor &r) {  // t:output, r:result tensor
 double e; J i,j; Scalar a,z,n; IntArrayRef s;
 bool b=xsize(x,1,s);  //true if size is given as 2nd arg (last arg is output tensor)
 switch(m) {
  case Tensormode::empty: if(b && x->n==3) r=torch::empty_out(t,s); break;
  case Tensormode::zeros: if(b && x->n==3) r=torch::zeros_out(t,s); break;
  case Tensormode::ones:  if(b && x->n==3) r=torch::ones_out(t,s); break;
  case Tensormode::rand:  if(b && x->n==3) r=torch::rand_out(t,s); break;
  case Tensormode::randn: if(b && x->n==3) r=torch::randn_out(t,s); break;
  case Tensormode::full:  if(b && x->n==4 && xscalar(x,2,a)) r=torch::full_out(t,s,a); break;
  case Tensormode::randperm: if (x->n==3 && xlong(x,1,i)) r=torch::randperm_out(t,i); break;
  case Tensormode::randint:
   b=xsize(x,x->n-2,s);
   if     (b && x->n==4 && xlong(x,1,j))                 r=torch::randint_out(t,j,s);
   else if(b && x->n==5 && xlong(x,1,i) && xlong(x,2,j)) r=torch::randint_out(t,i,j,s);
   break;
  case Tensormode::eye:
    if     (x->n==3 && xlong(x,1,i))                 r=torch::eye_out(t,i);
    else if(x->n==4 && xlong(x,1,i) && xlong(x,2,j)) r=torch::eye_out(t,i,j);
    break;
  case Tensormode::range:
  case Tensormode::arange:
   b=m==Tensormode::range;
   if     (x->n==3 && xnum(x,1,z))                              r = b ? torch::range_out(t,0,z)   : torch::arange_out(t,z);
   else if(x->n==4 && xnum(x,1,a) && xnum(x,2,z))               r = b ? torch::range_out(t,a,z)   : torch::arange_out(t,a,z);
   else if(x->n==5 && xnum(x,1,a) && xnum(x,2,z) && xnum(x,3,n))r = b ? torch::range_out(t,a,z,n) : torch::arange_out(t,a,z,n);
   break;
  case Tensormode::linspace:
  case Tensormode::logspace:
   b=m==Tensormode::logspace; i=100; e=10.0; //default of 100 steps, base 10
   if(xnum(x,1,a) && xnum(x,2,z) && (x->n==4 || (xlong(x,3,i) && (x->n==5 || (x->n==6 && b && xnum(x,4,e))))))
    r = b ? torch::logspace_out(t,a,z,i,e) : torch::linspace_out(t,a,z,i);
   break;
  case Tensormode::complex:
   if(x->n==3) {
    r=complex1(kput(x,1),t);
   } else if(x->n==4) {
    if(xbool(x,2,b))
     r=complex1(kput(x,1),b,t);
    else
     r=complex2(kput(x,1),kput(x,2),t);
   }
   break;
  case Tensormode::sparse:
   TORCH_ERROR("tensor: sparse not implemented with output tensors");
   break;
  default:
   TORCH_ERROR("tensor: unexpected tensor mode `",kK(x)[0]->s," supplied with output tensor");
   break;
 }
}

static void tensoropt(K x,Tensormode m,Tensor &r) {
 double e; J i,j,nx=x->n; Scalar a,z,n; IntArrayRef s; TensorOptions o;
 bool b=xopt(x,x->n-1,o); if(b) nx--;                         //track if options in last arg
 bool sz=xsize(x,1,s) && nx==((m==Tensormode::full) ? 3 : 2); //2nd arg is size & correct arg count
 checksparse(o,m); checkint(o,m);
 switch(m) {
  case Tensormode::empty: if(sz) r=torch::empty(s,o); break;
  case Tensormode::zeros: if(sz) r=torch::zeros(s,o); break;
  case Tensormode::ones:  if(sz) r=torch::ones(s,o); break;
  case Tensormode::rand:  if(sz) r=torch::rand(s,o); break;
  case Tensormode::randn: if(sz) r=torch::randn(s,o); break;
  case Tensormode::full:
   if(sz && xscalar(x,2,a))
    r=torch::full(s,a,o.has_dtype() ? o : o.dtype(maptype(kK(x)[2]->t)));
   break;
  case Tensormode::randperm:
   if(!o.has_dtype()) o=o.dtype(torch::kLong);
   if(nx==2 && xlong(x,1,i)) r = torch::randperm(i,o);
   break;
  case Tensormode::randint:
   sz=xsize(x,nx-1,s); // true if size is supplied as last non-options arg
   if(!o.has_dtype()) o=o.dtype(torch::kLong);
   if     (sz && nx==3 && xlong(x,1,j))                 r=torch::randint(j,s,o);
   else if(sz && nx==4 && xlong(x,1,i) && xlong(x,2,j)) r=torch::randint(i,j,s,o);
   break;
  case Tensormode::eye:
    if     (nx==2 && xlong(x,1,i))                 r=torch::eye(i,o);
    else if(nx==3 && xlong(x,1,i) && xlong(x,2,j)) r=torch::eye(i,j,o);
    break;
  case Tensormode::range:
   if     (nx==3 && xnum(x,1,a) && xnum(x,2,z))               r=torch::range(a,z,o);
   else if(nx==4 && xnum(x,1,a) && xnum(x,2,z) && xnum(x,3,n))r=torch::range(a,z,n,o);
   break;
  case Tensormode::arange:
   b=!o.has_dtype();
   if(nx==2 && xnum(x,1,z)) {
    if(b && z.isIntegral(false)) o=o.dtype(torch::kLong);
    r=torch::arange(z,o);
   } else if(nx==3 && xnum(x,1,a) && xnum(x,2,z)) {
    if(b && a.isIntegral(false) && z.isIntegral(false)) o=o.dtype(torch::kLong);
    r=torch::arange(a,z,o);
   } else if(nx==4 && xnum(x,1,a) && xnum(x,2,z) && xnum(x,3,n)) {
    if(b && a.isIntegral(false) && z.isIntegral(false) && n.isIntegral(false)) o=o.dtype(torch::kLong);
    r=torch::arange(a,z,n,o);
   }
   break;
  case Tensormode::linspace:
  case Tensormode::logspace:
   b=m==Tensormode::logspace; i=100; e=10.0; //default of 100 steps, base 10
   if(xnum(x,1,a) && xnum(x,2,z) && (nx==3 || (xlong(x,3,i) && (nx==4 || (nx==5 && b && xnum(x,4,e))))))
    r = b ? torch::logspace(a,z,i,e,o) : torch::linspace(a,z,i,o);
   break;
  case Tensormode::complex:
   if(nx==2) {
    r=complex1(kput(x,1), o);
   } else if(nx==3) {
    if(xbool(x,2,b))
     r=complex1(kput(x,1), o, b);
    else
     r=complex2(kput(x,1), kput(x,2), o);
   }
   break;
  case Tensormode::sparse:
   // as of version 1.8.1, sparse_coo_tensor seems to ignore tensor options if indices & values supplied
   // there is a check if explicit strided setting and gradient required throws nyi, requiring workaround below
   // https://github.com/pytorch/pytorch/issues/55453
   checksparse(o,m);
   if(nx==2 && xsize(x,1,s))
    r=torch::sparse_coo_tensor(s,o);
   else if(nx==3)
    r=to(torch::sparse_coo_tensor(kput(x,1),kput(x,2),o.requires_grad(false)), o);
   else if(nx==4 && xsize(x,3,s))
    r=to(torch::sparse_coo_tensor(kput(x,1),kput(x,2),s,o.requires_grad(false)), o);
   break;
  default: break;
 }
 // most tensor creation functions don't support newer memory format options yet (as of version 1.8.1)
 if(o.has_memory_format() && r.suggest_memory_format() != o.memory_format_opt().value()) {
  torch::NoGradGuard g;
  r=r.is_pinned() ? r.contiguous(*o.memory_format_opt()).pin_memory() : r.contiguous(*o.memory_format_opt());
  if(o.requires_grad()) r.set_requires_grad(true);
 }
}

static K tensormode(K x,S s,Tensormode m) {
 Tensor t,r; bool in=false,out=false;
 if((in=xten(x,1,t)))            tensorlike(x,m,t,r); // input tensor is 2nd arg
 else if((out=xten(x,x->n-1,t))) tensorout(x,m,t,r);  // output tensor is final arg
 else                            tensoropt(x,m,r);    // no input/output tensor
 TORCH_CHECK(r.defined(),"unrecognized argument(s) for tensor creation mode: ",s);
 return out ? (K)0 : kten(r);
}

static K tensorput(K x) {
 Tensor r,t; TensorOptions o;
 t=((xopt(x,1,o) || xten(x,1,r)) && x->n==2) ? kput(x,0) : kput(x);
 if(r.defined()) {
  r.resize_(t.sizes()).copy_(t,true);
  return (K)0;
 } else {
  //return kten(to(t,o));  // NULL TENSOR
  return kten(t.defined() ? to(t,o) : t);
 }
}

static Tensor tensorget(const Tensor& t,J d,K x) {   // d:dimension, x:index/indices
 if(x->t == -KJ)
  return t.select(d,x->j);
 else if(x->t == KJ)
  return torch::index_select(t,d,kput(x).to(t.device()));
 else
  TORCH_ERROR("tensor: last arg expected to be long(s) for indexing, given ",kname(x));
}

Tensor tensorget(const Tensor& t,K x) {
 bool b=false,c; J d=0; Tensor r;
 if((c=xbool(x,1,b)))
  TORCH_CHECK(t.is_complex(), "tensor: optional flag is only for complex tensors");
 if(x->n == 1+c) {                     // ptr or (ptr;flag)
  r=t;
 } else if(x->n == 2+c) {              // (ptr;ind) or (ptr;flag;ind)
  r=tensorget(t,d,kK(x)[x->n-1]);
 } else if(x->n == 3+c) {              // (ptr;dim;ind) or (ptr;flag;dim;ind)
  TORCH_CHECK(xlong(x,1+c,d), "tensor: ",(c ? "2nd" : "3rd")," arg of dimension expected as a long scalar, given ",kname(x,1+c));
  r=tensorget(t,d,kK(x)[x->n-1]);
 } else {
  TORCH_ERROR("tensor: up to ",3+c," args expected, (tensor;", (c ? "flag;" : ""),"dim;ind), but ",x->n," args given");
 }
 return c ? toreal(r,b) : r;
}

static K vectorptr(const TensorVector& v,K x) {
 if(x->n==1) {                                 // no additional args, return list of tensor ptrs
  J i=0; K r=ktn(0,v.size());
  for(const auto& t:v) kK(r)[i++]=kten(t);
  return r;
 } else if(x->n==2) {                          // 2nd arg of single index or list of indices
  K y=kK(x)[1];
  if(y->t == -KJ) {
   return kten(v.at(y->j));                              // single index, return tensor ptr
  } else if(y->t == KJ) {                                // indices, return list of selected tensor ptrs
   K r=ktn(0,y->n);
   for(J i=0; i<y->n;++i) kK(r)[i]=kten(v.at(kJ(y)[i]));
   return r;
  } else {
   TORCH_ERROR("tensor: given vector, 2nd arg expected to be long(s) for indexing, not ",kname(y));
  }
 } else {
  TORCH_ERROR("tensor: given vector, expecting no more than one additional indexing argument but given ",x->n-1," additional args");
 }
}

static K dictptr(const TensorDict& d,K x) {
 if(x->n==1) {                                       // no additional args, return k dict of tensor ptrs
  J i=0; K k=ktn(KS,d.size()),v=ktn(0,d.size());
  for(const auto &a:d) {
   kS(k)[i]=cs(a.key().c_str());
   kK(v)[i]=kten(a.value());
   ++i;
  }
  return xD(k,v);
 } else if(x->n==2) {                                 // additional indexing arg
  K y=kK(x)[1];
  if(y->t == -KS) {                                   // single symbol, return tensor pointer
   return kten(d[y->s]);
  } else if(y->t == KS) {                             // list of symbols, return k dict of selected tensor ptrs
    K r=ktn(0,y->n);
    for(J i=0; i<y->n;++i) kK(r)[i]=kten(d[kS(y)[i]]);
    return r;
  } else {
   TORCH_ERROR("tensor: given dictionary, 2nd arg expected to be symbols(s) for indexing, not ",kname(y));
  }
 } else {
  TORCH_ERROR("tensor: given dictionary, expecting no more than one additional indexing argument but given ",x->n-1," additional args");
 }
}

KAPI tensor(K x) {
 KTRY
  S s; Tensormode m; Ktag *g;
  if((g=xtag(x)) || (g=xtag(x,0))) {
   switch(g->a) {
    case Class::tensor: return kget(tensorget(((Kten*)g)->t, x));
    case Class::vector: return vectorptr(((Kvec*)g)->v, x);
    case Class::dict:   return  dictptr(((Kdict*)g)->d, x);
    default: TORCH_ERROR("tensor not implemented for ",mapclass(g->a));
   }
  } else if(xmode(x,0,s,m)) {
   return tensormode(x,s,m);
  } else {
   return tensorput(x);
  }
 KCATCH("tensor");
}

// ------------------------------------------------------------------------------------------
// tensor vector fns: 
// ------------------------------------------------------------------------------------------
// vec - initialize vector of tensors from k array, tensor ptr(s) or some mix of both
// vector - create vector of tensors, or return vector or vector element, or replace element
// dict - create dictionary of tensors, or return dictionary value(s)
// ------------------------------------------------------------------------------------------
TensorVector vec(K x,bool b) {   // b: true if any encountered tensor ptr to be de-referenced
 TensorVector v;
 if(x->t) {
  Tensor t=kput(x);
  if(t.dim())
   for(int64_t i=0;i<t.size(0);++i)
    v.emplace_back(t[i].clone());
  else
   v.emplace_back(t);
 } else if(xptr(x)) {
  if(kput(v,-1,x))
   if(b) kfree(x);
 } else {
  bool a=false;
  for(J i=0;i<x->n;++i)
   if(kput(v, -1, kK(x)[i])) a=true;
  if(a && b)
   for(J i=0;i<x->n;++i) if(xptr(x,i)) kfree(x,i);
 }
 return v;
}

KAPI vector(K x) {
 KTRY
  if(auto* v=xvec(x)) {             // if previously created vector, return as k list
   return kget(*v);
  } else if(auto* v=xvec(x,0)) {                 // if previously created vector
   if(x->n==2) {                                 // 2 args
    if(auto *w=xvec(x,1)) {                      // add additional vector
     for(size_t i=0,n=w->size(); i<n; ++i)
      v->emplace_back(w->at(i));         // add via index in case vector added to self
     if(v!=w) kfree(x,1);                // free vector added unless same
     return (K)0;
    } else {                            // else index into vector via 2nd arg of index/indices
     return kget(*v,kK(x)[1]);
    }
   } else if(x->n==3) {                 // if indices and values supplied
    kput(*v, kK(x)[1], kK(x)[2]);
    return (K)0;
   } else {
    TORCH_ERROR("vector: given ptr, expecting indices, vector ptr or (indices;values), but given ",x->n-1," additional arg(s)");
   }
  } else {
   return kvec(vec(x,true));
  }
 KCATCH("vector");
}

KAPI dict(K x) {
 KTRY
  TensorDict *d=xtensordict(x); if(!d) d=xtensordict(x,0);
  TORCH_CHECK(x->t==0 || x->t==99, "dict: not implemented for ",kname(x));
  if(d) {
   if(x->n==1) {                            // dict ptr
    return kget(*d);                        // return dictionary of syms!values to k
   } else if(x->n==2) {
    if(xdict(x,1))                          // dict(ptr;kdict)
     return kput(*d, kK(kK(x)[1])[0], kK(kK(x)[1])[1]), (K)0;
    else                                    // dict(ptr;sym(s))
     return kget(*d, kK(x)[1]);
   } else if(x->n==3) { 
    return kput(*d,kK(x)[1],kK(x)[2]), (K)0;
   } else {
    TORCH_ERROR("dict: expecting 1-3 args, but ",x->n," args supplied, not one of ptr, (ptr;dict), (ptr;syms), (ptr;syms;vals)");
   }
  } else {
   return kdict(kputd(x));
  }
 KCATCH("dict");
}

// --------------------------------------------------------------------------------------
//  complex tensors
// --------------------------------------------------------------------------------------
// kreal - handle api calls to extract real & imaginary parts of tensor
// real,imag - return real & imaginary parts of tensor as tensor or k value
// isreal - return boolean with 1's where value is real
// --------------------------------------------------------------------------------------
static K kreal(K x,Tensor(f)(const Tensor&),const char* nm) {
 KTRY
  bool b=false; Tensor *t=xten(x);
  if(!t) b=true, t=xten(x,0);  // enlisted tensor, return as k array
  TORCH_CHECK(t && t->is_complex() && x->n==1, nm, ": expects a complex tensor (enlist to return tensor ptr), given ",kname(x));
  // as of version 1.8.1, complex sparse tensors must  be made sparse real -> dense -> back to complex
  return kresult(b,f(t->is_sparse() ? torch::view_as_complex(sparsereal(*t).to_dense()) : *t));
 KCATCH(nm);
}

KAPI   real(K x) {return kreal(x, torch::real,   "real");}
KAPI   imag(K x) {return kreal(x, torch::imag,   "imag");}
KAPI isreal(K x) {return kreal(x, torch::isreal, "isreal");}

// ------------------------------------------------------------------------------------------
//  sparse tensors
// ------------------------------------------------------------------------------------------
// getsparse - handle api calls to extract indices & values from sparse tensor
// coalesce - colaesce a sparse tensor
// dense - return dense tensor given sparse tensor
// sparse - return sparse tensor given array, tensor, (array/tensor;dim), (array/tensor;mask)
// sparseindex - return non-zero indices of input as matrix/tensor, allow sparse dimension
// ------------------------------------------------------------------------------------------
static K getsparse(K x,bool i,const char* nm) {
 KTRY
  bool b=false; Tensor *t=xten(x);
  if(!t) b=true, t=xten(x,0);  // enlisted tensor, return as k array
  TORCH_CHECK(t && t->is_sparse() && x->n==1, nm, ": expects a sparse tensor (enlist to return tensor ptr), given ",kname(x));
  return kresult(b, i ? t->_indices() : t->_values());
 KCATCH(nm);
}

KAPI indices(K x) {return getsparse(x, true,  "indices");}
KAPI  values(K x) {return getsparse(x, false, "values");}

KAPI coalesce(K x) {
 KTRY
  Tensor *t=xten(x);
  TORCH_CHECK(t && t->is_sparse(), "coalesce: expecting sparse tensor, given ",kname(x));
  if(!t->is_coalesced())
   ((Kten*)xtag(x))->t=t->coalesce();
  return (K)0;
 KCATCH("coalesce");
}

KAPI dense(K x) {
 KTRY
  Tensor *t=xten(x);
  TORCH_CHECK(t && t->is_sparse(), "dense: expecting sparse tensor, given ",kname(x));
  return kten(t->to_dense());
 KCATCH("dense");
}

KAPI sparse(K x) {
 KTRY
  int64_t d; Tensor *i=nullptr,*t=xten(x);
  if(t) {
   return kten(t->to_sparse());
  } else if (!x->t && x->n==2 && (xint64(x,1,d) || ((i=xten(x,1))))) {
   t=xten(x,0);
   if(i) {
    TORCH_CHECK(i->is_sparse(), "sparse: 2nd arg is not a sparse tensor, unable to use its indices");
    return kten((t ? *t : kput(x,0)).sparse_mask(*i));
   } else {
    return kten((t ? *t : kput(x,0)).to_sparse(d));
   }
  } else if(xmixed(x,2)) {
    TORCH_ERROR("sparse: unrecognized arg(s), expecting input, (input;sparsedim) or (input;sparse tensor)");
  } else {
   return kten(kput(x).to_sparse());
  }
 KCATCH("sparse");
}

KAPI sparseindex(K x) {
 KTRY
  int64_t d; Tensor *t;
  if((t=xten(x))) {
   return kten(t->nonzero().t());
  } else if(x->n==2 && xint64(x,1,d)) {
   t=xten(x,0);
   return kresult(t,std::get<0>(torch::unique_consecutive((t ? *t : kput(x,0)).nonzero().index_select(1,torch::arange(d)),false,false,0)).t());
  } else {
   return kget(kput(x).nonzero().t());
  }
 KCATCH("sparseindex");
}

// ----------------------------------------------------------------------------------------------
// kcat1 - check for tensor(s) at first level of a general list, e.g. cat(a;b)
// kcat2 - check for (inputs;dim) or (inputs;dim;out tensor) or (inputs;out tensor)
// kcat - handle args for cat/stack
// cat - join arrays or tensors along given dimension (dim 0 if none given)
// stack - join same-sized arrays along a new dimension (leading dim if none given)
// ----------------------------------------------------------------------------------------------
using Fld = Tensor  (*)(         TensorList, int64_t);
using Gld = Tensor& (*)(Tensor&, TensorList, int64_t);

bool kcat1(K x) {for(J i=0;i<x->n;++i) if(xten(x,i)) return true; return false;}

bool kcat2(K x,int64_t& d, Tensor& r) {
 if(!x->t && x->n>1 && !kK(x)[0]->t)
  return (xint64(x,1,d) && (x->n==2 || (x->n==3 && xten(x,2,r)))) || (x->n==2 && xten(x,1,r));
 else
  return false;
}

K kcat(K x,Fld f,Gld g,const char* s) {
 KTRY
  TORCH_CHECK(!x->t, s," not implemented for ",kname(x->t));
  int64_t d=0; Tensor r; TensorVector *v;
  if((v=xvec(x))){
   return kten(f(*v,d));
  } else if((v=xvec(x,0))) {
   TORCH_CHECK(kcat2(x,d,r), s," expects vector, (vector;dim), (vector;dim;out tensor) or (vector;out tensor)");
   return r.defined() ? (g(r,*v,d), (K)0) : kten(f(*v,d));
  } else if(kcat1(x)) {
   return kten(f(vec(x,false),d));
  } else if((x->n==1 && !kK(x)[0]->t) || kcat2(x,d,r)) {
   K y=kK(x)[0];
   return r.defined() ? (g(r,vec(y,false),d), (K)0) : kresult(!y->t && kcat1(y),f(vec(y,false),d));
  } else {
   return kget(f(vec(x,false),d));
  }
 KCATCH(s);
}

KAPI cat(K x)   {return kcat(x, torch::cat,   torch::cat_out,   "cat");}
KAPI stack(K x) {return kcat(x, torch::stack, torch::stack_out, "stack");}

// ----------------------------------------------------------------------------------------------
// permute - k api for permute (not just for complex tensors)
// expand - expand tensor or array given sizes or tensor w'size to copy
// squeeze/unsqueeze - remove or add dimension to input array/tensor, boolean in-place option
// ----------------------------------------------------------------------------------------------
KAPI permute(K x) {
 KTRY
  Tensor *t=xten(x,0); IntArrayRef n;
  TORCH_CHECK(!x->t && x->n==2, "permute: unexpected arg(s), expecting (input/tensor; reordered dimensions)");
  TORCH_CHECK(xsize(x,1,n), "permute: expecting 2nd arg of reordered dimensions, given ",kname(x,1));
  return kresult(t, (t ? *t : kput(x,0)).permute(n));
 KCATCH("permute");
}

KAPI expand(K x) {
 KTRY
  IntArrayRef n; Tensor *t=xten(x,0),*s=xten(x,1);
  if((s || xsize(x,1,n)) && x->n==2) {
   if(t)
    return kten(s ? t->expand_as(*s) : t->expand(n));
   else
    return kget(s ? kput(x,0).expand_as(*s) : kput(x,0).expand(n));
  } else if(xsize(x,n) && n.size()==2) {
    return kget(torch::scalar_to_tensor(n[0]).expand(n[1]));
  } else {
   TORCH_ERROR("expand expects (input array/tensor; size) or (input array/tensor; tensor w'size to match)");
  }
 KCATCH("expand");
}

K ksqueeze(K x,bool a,const char* s) {
 KTRY
  bool b=false; int64_t d=nj; Tensor *t;
  if((t=xten(x))) {
   if(a) return kten(t->squeeze());
   else  return KERR("unsqueeze requires 2nd arg specifying dimension to add");
  } else if((xint64(x,1,d) && (x->n==2 || (x->n==3 && xbool(x,2,b)))) || (xbool(x,1,b) && x->n==2)) {
   TORCH_CHECK(a || d != nj, s," requires 2nd arg specifying dimension to add");
   if((t=xten(x,0))) {
    if(!a)           return b ? t->unsqueeze_(d),(K)0 : kten(t->unsqueeze(d));
    else if(null(d)) return b ? t->squeeze_(),   (K)0 : kten(t->squeeze());
    else             return b ? t->squeeze_(d),  (K)0 : kten(t->squeeze(d));
   } else {
    if(!a)           return kget(kput(x,0).unsqueeze(d));
    else if(null(d)) return kget(kput(x,0).squeeze());
    else             return kget(kput(x,0).squeeze(d));
   }
  } else if(a) {
   return kget(kput(x).squeeze());
  } else {
   TORCH_ERROR(s, ": unexpected arg(s), expects (input;dim;optional in-place flag)");
  }
 KCATCH(s);
}

KAPI squeeze(K x)   {return ksqueeze(x, true,  "squeeze");}
KAPI unsqueeze(K x) {return ksqueeze(x, false, "unsqueeze");}

// ----------------------------------------------------------------------------------------------
// tensorlong - tensor/vector attributes returned to k as long scalar
// tensorsym - tensor/vector attributes returned to k as a symbol, e.g. device
// tensorflag - tensor/vector attributes returned to k as a boolean, e.g. leaf
// tensorsize - tensor/vector attributes returned to k as a long list, e.g. size/stride
// tensorattr - handle tensor attribute queries according to k datatype returned
// ----------------------------------------------------------------------------------------------
static J storlong(const Storage& s,Attr a) {
 switch(a) {
  case Attr::ptr:         return (intptr_t)s.data();
  case Attr::ref:         return s.use_count();
  default: TORCH_ERROR(mapattr(a),": not implemented for storage");
 }
}

J tensorlong(const Tensor& t,Attr a) {
 switch(a) {
  case Attr::dim:       return t.dim();
  case Attr::itemsize:  return t.is_sparse() ? tensorlong(t._values(),a) : t.dtype().itemsize();
  case Attr::numel:     return t.is_sparse() ? t._values().numel()       : t.numel();
  case Attr::elements:  return objnum(t);
  case Attr::bytes:     return objbytes(t);
  case Attr::offset:    return t.is_sparse() ? nj : t.storage_offset();
  case Attr::ref:       return t.use_count();
  case Attr::sref:      return t.defined() ? t.storage().use_count() : 0;
  case Attr::weakref:   return t.defined() ? t.weak_use_count() : 0;
  case Attr::ptr:       return (intptr_t)t.unsafeGetTensorImpl();
  case Attr::sptr:      return !t.defined() || t.is_sparse() ? 0 : (intptr_t)t.storage().data();
  case Attr::densedim:  return t.is_sparse() ? t.dense_dim()  : t.dim();
  case Attr::sparsedim: return t.is_sparse() ? t.sparse_dim() : 0;
  case Attr::nnz:       return t.is_sparse() ? t._nnz() : (t.defined() ? t.count_nonzero().item().toLong() : 0);
  default: TORCH_ERROR(mapattr(a),": not implemented for tensors");
 }
}

S tensorsym(const Tensor& t,Attr a) {
 switch(a) {
  case Attr::device:   return t.defined() ? optdev(t.device()) : nullsym();
  case Attr::dtype:    return t.defined() ? optdtype(t.dtype()) : nullsym();
  case Attr::layout:   return t.defined() ? optlayout(t.layout()) : nullsym();
  case Attr::gradient: return t.defined() ? optgrad(t.requires_grad()) : nullsym();
  case Attr::gradfn:   return (S)(t.defined() && t.grad_fn() ?  t.grad_fn()->name().c_str() : "");
  case Attr::pinned:   return optpin(!t.defined() || t.is_sparse() ? false : t.is_pinned());
  case Attr::memory:   return t.defined() ? optmemory(t.suggest_memory_format()) : nullsym();
  default: TORCH_ERROR(mapattr(a),": not implemented for tensors");
 }
}

static bool tensorflag(const Tensor &t,Attr a) {
 switch(a) {
  case Attr::coalesced:    return t.is_sparse() ? t.is_coalesced() : true;
  case Attr::contiguous:   return t.is_sparse() ? false : t.is_contiguous();
  case Attr::contiguous2d: return t.is_sparse() ? false : t.is_contiguous(torch::MemoryFormat::ChannelsLast);
  case Attr::contiguous3d: return t.is_sparse() ? false : t.is_contiguous(torch::MemoryFormat::ChannelsLast3d);
  case Attr::defined:      return t.defined();
  case Attr::gradflag:     return t.requires_grad();
  case Attr::leaf:         return !t.defined() || t.is_leaf();
  case Attr::pinned:       return !t.defined() || t.is_sparse() ? false : t.is_pinned();
  case Attr::sparseflag:   return t.is_sparse();
  default: TORCH_ERROR(mapattr(a),": not implemented for tensors");
 }
}

K tensorsize(const Tensor &t,Attr a) {
 switch(a) {
  case Attr::size:    return klist(t.dim(),t.sizes().data());
  case Attr::stride:  return !t.defined() || t.is_sparse() ? ktn(0,0) : klist(t.dim(),t.strides().data());
  default: TORCH_ERROR(mapattr(a),": not implemented for tensors");
 }
}

K tensorattr(const Tensor &t,Ktype k,Attr a) {
 switch(k) {
  case -KJ: return kj(tensorlong(t,a));
  case  KJ: return tensorsize(t,a);
  case -KS: return ks(tensorsym(t,a));
  case -KB: return kb(tensorflag(t,a));
  default: TORCH_ERROR(mapattr(a),": not implemented for tensors");
 }
}

// ----------------------------------------------------------------------------------------------
// vattr - handle vector/dictionary values or other iterable to extract attributes -> k list
// vectorattr - handle tensor vector attribute queries according to k datatype returned
// dictattr - handle tensor dictionary queries, return dictionary of attribute values
// options - return dictionary/table of tensor/vector/dictionary attributes
// ----------------------------------------------------------------------------------------------
template<typename V> static K vattr(const V &v,Ktype k,Attr a) {
 size_t i=0; K x=ktn(k<0 ? abs(k) : 0, v.size());
 try {
  for(const auto& t:v) {
   switch(k) {
    case -KJ: kJ(x)[i]=tensorlong(t,a); break;
    case  KJ: kK(x)[i]=tensorsize(t,a); break;
    case -KS: kS(x)[i]=tensorsym(t,a);  break;
    case -KB: kG(x)[i]=tensorflag(t,a); break;
    default: TORCH_ERROR(mapattr(a),": not implemented for tensors");
   }
   ++i;
  }
 } catch(...) {
  if(x) r0(x);
  throw;
 }
 return x;
}

K vectorattr(const TensorVector &v,Ktype k,Attr a) {
 return vattr(v,k,a);
}

K dictattr(const TensorDict& d,Ktype k,Attr a) {
 K y=vattr(d.values(),k,a);
 J i=0; K x=ktn(KS,d.size());
 for(const auto& s:d.keys()) kS(x)[i++]=cs(s.c_str());
 return xD(x,y);
}

KAPI options(K x) {
 KTRY
  if(xempty(x)) {
   return optmap(TensorOptions());
  } else if(auto *t=xten(x)) {
   return optmap(*t);
  } else if(auto* v=xvec(x)) {
   K k=optkey(); K y=ktn(0,k->n);
   for(J i=0; i<k->n; ++i) 
    kK(y)[i]=ktn(KS,v->size());
   for(size_t i=0; i<v->size(); ++i)
    optval(v->at(i),y,i);
   return xT(xD(k,y));
  } else if(auto* d=xtensordict(x)) {
   K c=optkey(); K k=ktn(KS,d->size()),y=ktn(0,c->n); J i;
   for(i=0; i<y->n; ++i) kK(y)[i]=ktn(KS,d->size());
   i=0;
   for(const auto& a:*d) {
    kS(k)[i]=cs(a.key().c_str());
    optval(a.value(),y,i++);
   }
   return xD(k,xT(xD(c,y)));
  } else {
   TORCH_ERROR("options: unrecognized arg(s), expected empty arg for defaults or tensor,vector or dictionary of tensors");
  }
 KCATCH("options");
}

// ------------------------------------------------------------------------------------------------
// diagnostic functions -- check underlying pointers, storage data, reference counts, etc.
// ------------------------------------------------------------------------------------------------
// stordata - return storage data as k list (first move from cuda -> cpu, convert half to float)
// storinfo - return storage attributes & data as dictionary
// tensorinfo - return dictionary of attributes given tensor and detail flag
// ------------------------------------------------------------------------------------------------
static K stordata(const Tensor& a) {
 const auto& t=(a.dtype()==torch::kHalf) ? a.cpu().to(torch::kFloat) : a.cpu();
 const auto& d=t.dtype();
 const auto& s=t.storage();
 K x=ktn(maptype(d),s.nbytes() / d.itemsize());
 memcpy(kG(x),s.data(),s.nbytes());
 return x;
}

static K storinfo(const Tensor& t) {
 K x=xD(ktn(KS,0),ktn(0,0)),*k=&kK(x)[0],*v=&kK(x)[1];
 js(k, mapattr(Attr::bytes));    jk(v, kj(t.defined() ? t.storage().nbytes() : 0));
 js(k, mapattr(Attr::ref));      jk(v, kj(t.defined() ? storlong(t.storage(), Attr::ref) : 0));
 js(k, mapattr(Attr::ptr));      jk(v, kj(t.defined() ? storlong(t.storage(), Attr::ptr) : 0));
 js(k, mapattr(Attr::data));     jk(v, t.defined() ? stordata(t) : knk(0,0));
 return x;
}

K tensorinfo(const Tensor& t,bool d) {
 if(d && t.is_sparse()) {
  K x=xD(ktn(KS,0),ktn(0,0)),*a=&kK(x)[0],*b=&kK(x)[1];
  js(a, cs("indices")); jk(b, tensorinfo(t._indices(),d));
  js(a, cs("values"));  jk(b, tensorinfo(t._values(),d));
  return x;
 }
 K x=xD(ktn(KS,0),ktn(0,0)),*k=&kK(x)[0],*v=&kK(x)[1]; bool b=t.defined();
 js(k, mapattr(Attr::defined));     jk(v, kb(tensorflag(t, Attr::defined)));
 js(k, mapattr(Attr::device));      jk(v, ks(b ? tensorsym(t,Attr::device) : nullsym()));
 js(k, mapattr(Attr::dtype));       jk(v, ks(b ? tensorsym(t,Attr::dtype) : nullsym()));
 js(k, mapattr(Attr::layout));      jk(v, ks(b ? tensorsym(t,Attr::layout) : nullsym()));
 js(k, mapattr(Attr::gradient));    jk(v, ks(b ? tensorsym(t,Attr::gradient) : nullsym()));
 js(k, mapattr(Attr::pinned));      jk(v, ks(b ? tensorsym(t,Attr::pinned) : nullsym()));
 js(k, mapattr(Attr::memory));      jk(v, ks(b ? tensorsym(t,Attr::memory) : nullsym()));
 js(k, mapattr(Attr::leaf));        jk(v, kb(tensorflag(t, Attr::leaf)));
 js(k, mapattr(Attr::gradfn));      jk(v, ks(tensorsym(t,  Attr::gradfn)));
 js(k, mapattr(Attr::dim));         jk(v, kj(tensorlong(t, Attr::dim)));
 js(k, mapattr(Attr::sparsedim));   jk(v, kj(tensorlong(t, Attr::sparsedim)));
 js(k, mapattr(Attr::size));        jk(v, tensorsize(t,    Attr::size));
 js(k, mapattr(Attr::stride));      jk(v, tensorsize(t,    Attr::stride));
 js(k, mapattr(Attr::numel));       jk(v, kj(tensorlong(t, Attr::numel)));
 js(k, mapattr(Attr::itemsize));    jk(v, kj(tensorlong(t, Attr::itemsize)));
 js(k, mapattr(Attr::contiguous));  jk(v, kb(tensorflag(t, Attr::contiguous)));
 js(k, mapattr(Attr::coalesced));   jk(v, kb(tensorflag(t, Attr::coalesced)));
 js(k, mapattr(Attr::offset));      jk(v, kj(tensorlong(t, Attr::offset)));
 js(k, mapattr(Attr::ptr));         jk(v, kj(tensorlong(t, Attr::ptr)));
 js(k, mapattr(Attr::ref));         jk(v, kj(tensorlong(t, Attr::ref)));
 if(d) {
  js(k, mapattr(Attr::storage));   
  jk(v, storinfo(t));
 }
 return x;
}

// --------------------------------------------------------------------------------------------
// perm - return permutation indices given tensor and dimension
// vcheck - check for matching dimension size & device for each tensor in vector
// vperm - check vector for same dim size & device, return permutation indices
// shuffle - shuffle tensor or vector of same size along given dimension
// shuffle_ - in-place version of tensor/vector shuffle
// kshuffle,1,2 - k api functions, expects tensor/vector input or (input;dim;inplace flag)
// --------------------------------------------------------------------------------------------
static Tensor perm(const Tensor& t,int64_t d) {
 return torch::randperm(t.size(d),torch::dtype(torch::kLong).device(t.device()));
}

static void vcheck(const TensorVector& v,int64_t d) {
 int64_t i=0,n; Device c=torch::kCPU;
 for(const auto& t:v) {
  if(!i)
   n=t.size(d),c=t.device();
  else if(n != t.size(d))
   TORCH_ERROR("size mismatch: tensor[",i,"] size=",t.size(d),", but previous tensor(s) have size=",n," for dim ",d);
  else if (c != t.device())
   TORCH_ERROR("device mismatch: tensor[",i,"] is on ",t.device(),", but previous tensor(s) are on ", c);
  ++i;
 }
}

static Tensor vperm(const TensorVector& v,int64_t d) {vcheck(v,d); return v.size() ? perm(v[0],d) : Tensor();}

Tensor shuffle(const Tensor &t,int64_t d) {return t.index_select(d,perm(t,d));}
void shuffle_(Tensor &t,int64_t d) {t=shuffle(t,d);}

TensorVector shuffle(const TensorVector& v,int64_t d) {
 auto p=vperm(v,d); TensorVector r;
 for(const auto& t:v) r.emplace_back(t.index_select(d,p));
 return r;
}
 
void shuffle_(TensorVector& v,int64_t d) {
 auto p=vperm(v,d);
 for(auto& t:v) t=t.index_select(d,p);
}

static K kshuffle1(Tensor &t,int64_t d,bool b) {return b ? shuffle_(t,d),(K)0 : kten(shuffle(t,d));}
static K kshuffle2(TensorVector& v,int64_t d,bool b) {return b ? shuffle_(v,d),(K)0 : kvec(shuffle(v,d));}

KAPI kshuffle(K x) {
 KTRY
  bool b=true; int64_t d=0; Ktag *g; //default is in-place, along 1st dim
  if((g=xtag(x)) || 
    ((g=xtag(x,0)) && ((x->n==2 && (xint64(x,1,d) || xbool(x,1,b))) ||
                       (x->n==3 &&  xint64(x,1,d) && xbool(x,2,b)))))
   switch(g->a) {
    case Class::tensor: return kshuffle1(((Kten*)g)->t,d,b);
    case Class::vector: return kshuffle2(((Kvec*)g)->v,d,b);
    default: TORCH_ERROR("shuffle not implemented for ",mapclass(g->a));
   }
  else
   TORCH_ERROR("unrecognized arg(s) for shuffle");
 KCATCH("shuffle");
}

// -------------------------------------------------------------------------------------------
// subsets - given subset size, total size & optional drop last flag, return number of subsets
//  subset - take a subset of a particular tensor dimension, given offset & width
//           operate on vector of tensors if same size on given dimension
//           requires tensor, dim, offset, width
//           optional max size of given dim can be supplied, else calculated from storage size
// setsafe - calls set_() after checking that the length implied by sizes & strides will fit
// subsetsafe - alternate form of subset using setsafe rather than maximum size dimension
// reset - [re]set tensor offset, size & storage
// -------------------------------------------------------------------------------------------
int64_t subsets(int64_t w,int64_t n,bool b) {return n%w ? n/w + !b : n/w;}

void subset(Tensor& t,int64_t d,int64_t i,int64_t w,int64_t n) {
 if(!n) n=maxsize(t,d);  // if not given, get max size of dimension d from overall storage size
 TORCH_CHECK(i<n,"subset offset of ",i," must be from 0-",n-1," the maximum size for dimension ",d);
 if(w>n) w=n;            // reduce subset window if greater than max size
 if(w>n-i) w=n-i;        // final subset may be a fraction of window
 t.set_(t.storage(), i*t.stride(d), w==t.size(d) ? t.sizes() : newsize(t,d,w), t.strides());
}

void subset(TensorVector& v,int64_t d,int64_t i,int64_t w,int64_t n) {
 if(!n) n=maxsize(v,d);
 for(auto& t:v) subset(t,d,i,w,n);
}

void setsafe(Tensor& t,int64_t i,const IntArrayRef& sz,const IntArrayRef& st) {
 const Storage& s=t.storage();
 TORCH_CHECK(s.nbytes()>=i+at::detail::computeStorageNbytes(sz,st,t.dtype().itemsize()), 
            "size ",sz," and stride ",st," require total of ",
             at::detail::computeStorageNbytes(sz,st,1),
            " plus offset of ",i," exceeds storage size of ",s.nbytes()/t.dtype().itemsize());
 t.set_(s,i,sz,st);
}

void subsetsafe(Tensor& t,int64_t d,int64_t i,int64_t w) {
 setsafe(t, i*t.stride(d), newsize(t,d,w), t.strides());
}

KAPI reset(K x) {
 KTRY
  Tensor t; J i;IntArrayRef y,z;
  TORCH_CHECK(xten(x,0,t), "reset: tensor expected as 1st argument");
  TORCH_CHECK(xlong(x,1,i), "reset: offset (long) expected as 2nd argument, given ",kname(x,1));
  TORCH_CHECK(x->n<3 || xsize(x,2,y), "reset: size(s) expected as 3rd argument, given ",kname(x,2));
  TORCH_CHECK(x->n<4 || xsize(x,3,z), "reset: stride(s) expected as 4th argument, given ",kname(x,3));
  TORCH_CHECK(x->n>1 && x->n<5, "reset: up to 4 arguments expected, (tensor;offset;sizes;strides), but ",x->n," given");
  if(x->n==2)
   setsafe(t,i,t.sizes(),t.strides());            // existing size & stride, just move offset
  else if(x->n==3)
   setsafe(t,i,y,at::detail::defaultStrides(y));  // derive stride from size
  else
   setsafe(t,i,y,z);                              // specify offset, size & stride
  return (K)0;
 KCATCH("reset");
}

// -------------------------------------------------------------------------------------------
// batch - k api function to batch tensor/vector in place
// restore - k api function to restore tensor/vector to full size
// -------------------------------------------------------------------------------------------
KAPI batch(K x) {
 KTRY
  Tensor *t=nullptr; TensorVector *v=nullptr; int64_t i,w,n;
  TORCH_CHECK((t=xten(x,0))||(v=xvec(x,0)), "batch expects 1st arg of tensor or vector");
  TORCH_CHECK(x->n==3, "batch expects 3 args, (tensor/vector; batch index; batch size)");
  TORCH_CHECK(xint64(x,1,i) && i>-1, "batch: 2nd arg is not a valid batch index");
  TORCH_CHECK(xint64(x,2,w) && w> 0, "batch: 3rd arg is not a valid batch size");
  n=t ? maxsize(*t,0) : maxsize(*v,0);
  TORCH_CHECK(i*w<n, "batch index of ",i," beyond max of ",subsets(w,n)-1);
  if(t)
   subset(*t,0,i*w,w,n);
  else
   subset(*v,0,i*w,w,n);
  return (K)0;
 KCATCH("batch")
}

KAPI restore(K x) {
 KTRY
  Tensor *t=xten(x); TensorVector *v=xvec(x);
  TORCH_CHECK(t||v, "restore expects tensor or vector");
  if(t) fullsize(*t,0); else fullsize(*v,0);
  return (K)0;
 KCATCH("restore")
}

// -------------------------------------------------------------------------------------------
// narrow - narrow a tensor/vector along given dimension, according to offset,size
//          in-place flag defaults to false
// -------------------------------------------------------------------------------------------
KAPI narrow(K x) {
 KTRY
  bool b=false; int64_t d,i,w;
  TORCH_CHECK(xint64(x,1,d) && xint64(x,2,i) && xint64(x,3,w) && (x->n==4 || (x->n==5 && xbool(x,4,b))),
             "narrow: unrecognized arg(s), expecting (array/tensor/vector; dim; offset; size; optional in-place flag)");
  if(auto *v=xvec(x,0)) {
   if(b) {
    subset(*v,d,i,w);
    return (K)0;
   } else {
    TensorVector r;
    for(const auto &t:*v) r.emplace_back(t.narrow(d,i,w));
    return kvec(r);
   }
  } else if(auto *t=xten(x,0)) {
   return b ? subset(*t,d,i,w), (K)0 : kten(t->narrow(d,i,w));
  } else {
   return kget(kput(x,0).narrow(d,i,w));
  }
 KCATCH("narrow");
}

// -------------------------------------------------------------------------------------------
// transpose tensor/array, 3-d & higher require two dim's to swap, optional in-place flag
// -------------------------------------------------------------------------------------------
KAPI transpose(K x) {
 KTRY
  if(x->t) return r1(x); // k scalar/simple list returned as is
  bool b=false; J n=x->n-xbool(x,x->n-1,b); int64_t i,j; Tensor t;
  bool p=xten(x,t) || xten(x,0,t);
  if(!p) t=xmixed(x,4) ? kput(x,0) : (n=1,kput(x));
  TORCH_CHECK(n==3 || t.dim()<3, "transpose of ",t.dim(),"d tensor needs the two dimenstions to be swapped");
  if(n==1)
   return p ? (b ? t.t_(),(K)0 : kten(t.t())) : kget(t.t());
  else if(xint64(x,1,i) && xint64(x,2,j) && n==3)
   return p ? (b ? t.transpose_(i,j),(K)0 : kten(t.transpose(i,j))) : kget(t.transpose(i,j));
  else
   TORCH_ERROR("transpose expects tensor or (tensor;inplace flag) or (tensor;dim1;dim2;optional inplace flag");
 KCATCH("transpose");
}

// -------------------------------------------------------------------------------------------
// inferflag - return true if a dimension needs to be derived (resize_ does not infer)
// infersize - if one dim is -1, infer size from other sizes and overall number of elements
// kresize1 - handle different methods for resizing, also allow inferred dim for resize_
// kresize2 - handle inputs for resize/View/reshape
// resize - resize tensor in place given size or tensor w'size to use, also allow k array input
//          will reallocate storage if larger size required, elements will be uninitialized
//          inferflag/infersize used to allow resize_() to infer a dimension
// view - attempt to create a new tensor that is view of existing tensor (shares storage)
//        error if view size is not compatible with input tensor's size and stride 
// reshape - like view, but will create new storage for new tensor if view not possible
// -------------------------------------------------------------------------------------------
static bool inferflag(const IntArrayRef& s) {for(auto i:s) if(i<0)return true; return false;}

static std::vector<int64_t> infersize(IntArrayRef s, int64_t m) {
  int64_t n=1; auto r=s.vec(); auto i=c10::optional<int64_t>();
  for (int64_t d=0, ndim = s.size(); d != ndim; d++) {
    if (s[d] == -1) {
      if (i)
        throw std::runtime_error("only one dimension can be inferred");
      i=d;
    } else if(s[d] >= 0) {
      n *= s[d];
    } else {
      TORCH_ERROR("invalid shape dimension ", s[d]);
    }
  }
  if (m == n || (i && n > 0 && m % n == 0)) {
    if (i) {
      TORCH_CHECK(n != 0, "cannot reshape tensor of 0 elements into shape ", s,
                  " because the unspecified dimension size -1 can be any value and is ambiguous");
      r[*i] = m / n;
    }
    return r;
  }
  TORCH_ERROR("shape ",s," is invalid for input of size ",m);
}

static K kresize1(I m,bool p,Tensor&& t, const IntArrayRef& s) {
 switch(m) {
  case 0:  t.resize_(inferflag(s) ? infersize(s,t.numel()) : s);
           return p ? (K)0 : kget(t);
  case 1:  return kresult(p,t.reshape(s));
  case 2:  return kresult(p,t.view(s));
  default: return KERR("invalid resize/reshape mode");
 }
}

static K kresize2(K x,I m,const char* e) {
 KTRY
  IntArrayRef n; Tensor *t=xten(x,0),*s=xten(x,1);
  TORCH_CHECK(!x->t,   e," not implemented for ",kname(x->t));
  TORCH_CHECK(x->n==2, e," expects (array/tensor;new size/tensor w'size to use)");
  TORCH_CHECK(s || xsize(x,1,n), e," expects 2nd arg of size or tensor w'size to use");
  return kresize1(m, t, t ? *t : kput(x,0), s ? s->sizes() : n);
 KCATCH(e);
}

KAPI resize(K x)  {return kresize2(x, 0, "resize");}
KAPI reshape(K x) {return kresize2(x, 1, "reshape");}
KAPI view(K x)    {return kresize2(x, 2, "view");}

// -------------------------------------------------------------------------------------------
// newsize - return new vector for tensor sizes, replacing size at dimension d with new value
// maxsize - find the maximum size at given dimension using underlying storage size
// fullsize -  restore tensor(s) to maximum size at given dimension
// -------------------------------------------------------------------------------------------
std::vector<int64_t> newsize(const Tensor& t,int64_t d,int64_t n) {
 auto v=t.sizes().vec(); v.at(d)=n; return v;
}

int64_t maxsize(const Tensor& t,int64_t d) {
 int64_t n=1;
 for(auto i:t.sizes()) n*=i;
 return t.size(d)*t.storage().nbytes()/(n*t.dtype().itemsize());
}

int64_t maxsize(const TensorVector& v,int64_t d) {
 int64_t i=0,m=-1;
 for(const auto&t:v) {
  if(i) {
   auto n=maxsize(t,d);
   TORCH_CHECK(m==n, "tensor[",i,"] size=",n,", but previous tensor(s) have size=",m," for dim ",d);
  } else {
   m=maxsize(t,d);
  }
  ++i;
 }
 return m;
}

int64_t fullsize(Tensor& t,int64_t d) {
 auto m=maxsize(t,d);
 if(t.size(d) != m)
  t.set_(t.storage(), 0, newsize(t,d,m), t.strides());
 return m;
}

int64_t fullsize(TensorVector& v,int64_t d) {
 auto m=maxsize(v,d);
 for(auto& t:v) if(t.size(d) != m) fullsize(t,d);
 return m;
}

// ------------------------------------------------------------------------------------------
// zero - zero out tensor in place (if array, array-> tensor -> zero out -> return array)
// fill - fill  tensor/array with given element
// filldiagonal - fill diagonal of matrix input with given value
// ------------------------------------------------------------------------------------------
KAPI zero(K x) {
 KTRY
  if(auto* t=xten(x))
   return t->zero_(), (K)0;
  else
   return kget(kput(x).zero_());
 KCATCH("zero");
}

KAPI fill(K x) {
 KTRY
  Scalar s;
  if(xnum(x,1,s) && x->n==2)
   if(auto *t=xten(x,0))
    return t->fill_(s), (K)0;
   else
    return kget(kput(x,0).fill_(s));
  else
   TORCH_ERROR("fill expects (tensor/array;fill element)");
 KCATCH("fill");
}

KAPI filldiagonal(K x) {
 KTRY
  bool w=false; Scalar s;
  if(xnum(x,1,s) && (x->n==2 || (x->n==3 && xbool(x,2,w))))
   if(auto *t=xten(x,0))
    return t->fill_diagonal_(s,w), (K)0;
   else
    return kget(kput(x,0).fill_diagonal_(s,w));
  else
   TORCH_ERROR("fill diagonal expects (tensor/array;fill element;optional wrap flag)");
 KCATCH("fill diagonal");
}

// ------------------------------------------------------------------------------------------
// imagegrid - rearrange input images into a single tensor w'images across given cols & rows
// makegrid - api fn to accept array/tensor and rows,cols,padding,pad value -> grid of images
// ------------------------------------------------------------------------------------------
Tensor imagegrid(const Tensor& a,int64_t r,int64_t c,int64_t p,short v) {
 Tensor t=a;
 if(t.dim()==2) t=t.unsqueeze(0);           // single image H x W
 if(t.dim()==3) {                           // single image C x H x W
  if(t.size(0)==1) t=torch::cat({t,t,t},0); // if single-channel, convert to 3-channel
  t=t.unsqueeze(0);
 }
 if(t.dim()==4 && t.size(1)==1)
  t=torch::cat({t,t,t},1);                  // n single-channel images
 int64_t n=t.size(0),h=t.size(2)+p,w=t.size(3)+p,i,j,k=0;
 auto g=t.new_full({t.size(1),r*h+p,c*w+p},v);
 for(i=0; i<r; ++i) {
  for(j=0; j<c; ++j) {
   if(k>=n) break;
   g.narrow(1,i*h+p,h-p).narrow(2,j*w+p,w-p).copy_(t[k++]);
  }
 }
 return g;
}

KAPI makegrid(K x,K y,K z) {
 KTRY
  bool b; int64_t r=nj,c=nj,p=2,v=0; Tensor t;
  if(xint64(x,1,r) && (x->n==2 || (xint64(x,2,c) && (x->n==3 || (xint64(x,3,p) && (x->n==4 || (xint64(x,4,v) && x->n==5))))))) {
   if(!(b=xten(x,0,t))) t=kput(x,0);
   TORCH_CHECK(t.dim()==3 || t.dim()==4, "makegrid: expecting 3 or 4-dimensional input, ",t.dim()," dimension(s) given");
   TORCH_CHECK(r !=nj || c !=nj, "makegrid: both rows & columns cannot be null");
   if(t.dim()==3) t=t.unsqueeze(1);
   return kresult(b,imagegrid(t,r,c,p,v));
  } else {
   TORCH_ERROR("makegrid: unrecognized arg(s), expecting 2-5 args, (input array/tensor; rows; cols; padding; pad value)");
  }
 KCATCH("makegrid");
}

// ------------------------------------------------------------------------------------------
// tensorcopy - tgt <- src values, must have same type & device, tgt resized if src larger
// tensorcopy_ - copy in place method
// ------------------------------------------------------------------------------------------
void tensorcopy(Tensor &t,const Tensor &s,bool a) {
 if(s.dtype() != t.dtype()) {
  TORCH_ERROR("unable to copy values from ",s.dtype()," tensor to ",t.dtype()," tensor");
 } else if(s.device() != t.device()) {
  TORCH_ERROR("unable to copy values across devices, from ",s.device()," to ",t.device());
 } else {
  t.resize_as_(s).copy_(s,a);
 }
}

KAPI tensorcopy_(K x) {
 KTRY
  bool a=false; Tensor *t=xten(x,0),*s=xten(x,1);
  TORCH_CHECK(t && (x->n==2 || (x->n==3 && xbool(x,2,a))), "copy expects (tensor;input;optional async flag)");
  t->copy_(s ? *s : kput(x,1), a);
  return (K)0;
 KCATCH("copy");
}

// ------------------------------------------------------------------------------------------
// kgrad - return gradient data or empty, if ptr enlisted, return gradient ptr (must free)
// gradflag - return requires gradient boolean (as opposed to sym of `grad/`nograd)
// ------------------------------------------------------------------------------------------
KAPI kgrad(K x) {
 KTRY
  bool p=false; Tensor t;
  if(xten(x,t) || (p=(xten(x,0,t) && x->n==1))) {
   if(p) return t.grad().defined() ? kten(t.grad()) : KERR("no gradient defined");
   else  return t.grad().defined() ? kget(t.grad()) : (K)0;
 } else {
  return KERR("unexpected arg(s) for gradient, expecting tensor (enlist to return gradient ptr)");
 }
 KCATCH("unable to get gradient");
}

KAPI gradflag(K x) {
 KTRY
  bool b; Ktag *k=xtag(x); if(!k) k=xtag(x,0);
  if(k && x->n==1) {
   return attr(x, -KB, Attr::gradflag);
  } else if(k && x->n==2 && xbool(x,1,b)) {
   switch(k->a) {
    case Class::tensor:
     ((Kten*)k)->t.set_requires_grad(b);
     break;
    case Class::vector: 
     for(auto& t:((Kvec*)k)->v) t.set_requires_grad(b);
     break;
    case Class::dict:
     for(auto& t:((Kdict*)k)->d.values()) t.set_requires_grad(b);
     break;
    default: TORCH_ERROR("gradflag: not implemented for ",mapclass(k->a));
   }
   return (K)0;
  } else {
   TORCH_ERROR("gradflag: unrecognized arg(s), expecting tensor/vector/dictionary and optional flag");
  }
 KCATCH("gradflag");
}

// --------------------------------------------------------------------------------------
// vecdetach - detach specified element(s) of vector of tensors
// dictdetach - detach specified element(s) of dictionary of tensors
// detach - k api fn to detach tensor/vector/dictonary  
//          tensor has optional inplace flag, vec/dict allows additional indices/keys
// --------------------------------------------------------------------------------------
static void vecdetach(TensorVector& v,J n,J *j) {
 J m=v.size();
 for(J i=0;i<n;++i) {
   J k=j[i];
   TORCH_CHECK(k>=0, "detach: vector[",k,"] invalid");
   TORCH_CHECK(k<m, "detach: vector[",k,"] out of bounds, index must be less than ",m);
   v[k].detach_();
 }
}

static void dictdetach(TensorDict& d,J n,S *s) {
 for(J i=0;i<n;++i) {
  S k=s[i];
  TORCH_CHECK(d.contains(k),"detach: no dictionary parameter named `",k);
  d[k].detach_();
 }
}

KAPI detach(K x) {
 KTRY
  Ktag *g=xtag(x); bool a=g,b=false;
  TORCH_CHECK(g||(g=xtag(x,0)), "detach: expecting tensor, vector or dictionary");
  switch(g->a) {
   case Class::tensor: {
    TORCH_CHECK(a || (x->n==2 && xbool(x,1,b)), "detach: expecting tensor or (tensor; inplace flag)");
    auto &t=((Kten*)g)->t;
    return b ? t.detach_(),(K)0 : kten(t.detach());
   }
   case Class::vector: {
    auto &v=((Kvec*)g)->v;
    if(a) {
     for(auto &t:v) t.detach_();
    } else {
     TORCH_CHECK(x->n==2, "detach: vector supplied, but unexpected number of additional args(",x->n-1,")");
     K y=kK(x)[1];
     TORCH_CHECK(y->t==-KJ || y->t==KJ, "detach: vector supplied, expected 2nd arg of indices, given ",kname(y));
     y->t==KJ ? vecdetach(v,y->n,kJ(y)) : vecdetach(v,1,&y->j);
    }
    return (K)0;
   }
   case Class::dict: {
    auto &d=((Kdict*)g)->d;
    if(a) {
     for(auto &a:d) a.value().detach_();
    } else {
     TORCH_CHECK(x->n==2, "detach: dictionary supplied, but unexpected number of additional args(",x->n-1,")");
     K y=kK(x)[1];
     TORCH_CHECK(y->t==-KS || y->t==KS, "detach: dictionary supplied, expected 2nd arg of symbol key(s), given ",kname(y));
     y->t==KS ? dictdetach(d,y->n,kS(y)) : dictdetach(d,1,&y->s);
    }
    return (K)0;
   }
   default: TORCH_ERROR("detach: not implemented for ",mapclass(g->a));
  }
  return (K)0;
 KCATCH("detach");
}

// -------------------------------------------------
// ksame - check if same tensor/storage
// same - check if underlying tensor pointers match
// alias - check if same underlying storage
// -------------------------------------------------
static K ksame(K x,bool s) {
 KTRY
  Tensor *a=xten(x,0),*b=xten(x,1);
  TORCH_CHECK(a && b && x->n==2, (s ? "same" : "alias"), ": expects two tensors");
  return kb(s ? a->is_same(*b) : a->is_alias_of(*b));
 KCATCH("same/alias");
}

KAPI same(K x)  {return ksame(x,true);}
KAPI alias(K x) {return ksame(x,false);}

// ----------------------------------
// tensor fns defined in k namespace
// ----------------------------------
void tensorfn(K x) {
 fn(x, "tensor",       KFN(tensor),        1);
 fn(x, "zero",         KFN(zero),          1);
 fn(x, "fill",         KFN(fill),          1);
 fn(x, "filldiagonal", KFN(filldiagonal),  1);
 fn(x, "makegrid",     KFN(makegrid),      1);
 fn(x, "copy",         KFN(tensorcopy_),   1);
 fn(x, "grad",         KFN(kgrad),         1);
 fn(x, "gradflag",     KFN(gradflag),      1);
 fn(x, "detach",       KFN(detach),        1);
 fn(x, "same",         KFN(same),          1);
 fn(x, "alias",        KFN(alias),         1);
 fn(x, "vector",       KFN(vector),        1);
 fn(x, "dict",         KFN(dict),          1);
 fn(x, "real",         KFN(real),          1);
 fn(x, "imag",         KFN(imag),          1);
 fn(x, "isreal",       KFN(isreal),        1);
 fn(x, "indices",      KFN(indices),       1);
 fn(x, "values",       KFN(values),        1);
 fn(x, "coalesce",     KFN(coalesce),      1);
 fn(x, "dense",        KFN(dense),         1);
 fn(x, "sparse",       KFN(sparse),        1);
 fn(x, "sparseindex",  KFN(sparseindex),   1);
 fn(x, "cat",          KFN(cat),           1);
 fn(x, "permute",      KFN(permute),       1);
 fn(x, "stack",        KFN(stack),         1);
 fn(x, "expand",       KFN(expand),        1);
 fn(x, "squeeze",      KFN(squeeze),       1);
 fn(x, "unsqueeze",    KFN(unsqueeze),     1);
 fn(x, "options",      KFN(options),       1);
 fn(x, "shuffle",      KFN(kshuffle),      1);
 fn(x, "reset",        KFN(reset),         1);
 fn(x, "batch",        KFN(batch),         1);
 fn(x, "restore",      KFN(restore),       1);
 fn(x, "narrow",       KFN(narrow),        1);
 fn(x, "transpose",    KFN(transpose),     1);
 fn(x, "resize",       KFN(resize),        1);
 fn(x, "reshape",      KFN(reshape),       1);
 fn(x, "View",         KFN(view),          1);
}
