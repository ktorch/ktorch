#include "ktorch.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// --------------------------------------------------------------------------------------------------
// krrbuf - copy msg to a buffer for signalling error to k
// dictadd - add an entry in a dictionary mapping symbol -> k value
// xind - true if i is valid index of k list (type=0)
// kptr - given void *, add to pointer list & return k list of one long scalar = (intptr_t)void *
// xptr - given k value, return true if enclosed scalar and in pointer set
// xtag - if enclosed integer ptr detected from k, return pointer to tag structure
// xhelp - check for single argument: `help, or 2 symbols, e.g. `help`conv2d
// --------------------------------------------------------------------------------------------------
S krrbuf(const char *s) {
 static C thread_local b[4096]; b[0]=0; 
 return strncat(b, s, sizeof(b)-1);
}

void dictadd(K x,       char* s, K v){K *k=kK(x); js(&k[0],cs(s)); jk(&k[1],v);}
void dictadd(K x, const char* s, K v){K *k=kK(x); js(&k[0],cs(s)); jk(&k[1],v);}

bool xind(K x,J i) {return !x->t && -1<i && i<x->n;}
bool xind(K x,J i,Ktype k) {return x->t==k && -1<i && i<x->n;}
K kptr(void *v){J j=(intptr_t)v; pointer().insert(j); return knk(1,kj(j));}

bool xptr(K x) {
 if(!x->t && x->n==1 && kK(x)[0]->t==-KJ) {
  TORCH_CHECK(pointer().count(kK(x)[0]->j), "stale pointer");
  return true;
 } else {
  return false;
 }
}

bool xptr(K x,J i) {return xind(x,i) && xptr(kK(x)[i]);}

Ktag* xtag(K x) {return xptr(x) ? (Ktag*)kK(x)[0]->j : nullptr;}
Ktag* xtag(K x,J i) {return xind(x,i) ? xtag(kK(x)[i]) : nullptr;}

bool xhelp(K x) {return x->t == -KS && x->s == env().help;}
bool xhelp(K x,S &s) {
 if(x->t==KS && x->n == 2 && kS(x)[0]==env().help)
  return s=kS(x)[1],true;
 else
  return false;
}

// ------------------------------------------------------------------------------------------
// null - true if null for given type
// match - return true if scalars match (check long/double value)
// kscalar - return k double/long from torch scalar
// ------------------------------------------------------------------------------------------
bool null(const char* x) { return !x || !strlen(x);}
bool null(const J x)     { return x == nj; }

bool match(const Scalar &x,const Scalar &y) {
 if(x.isIntegral(false)) {
  if(y.isIntegral(false))
   return x.toLong() == y.toLong();
  else if(y.isFloatingPoint())
   return x.toDouble() == y.toDouble();
 } else if(x.isFloatingPoint()) {
  if(y.isFloatingPoint() || y.isIntegral(false))
   return x.toDouble() == y.toDouble();
 }
 AT_ERROR("unexpected scalar type(s), neither integral or floating point, cannot compare");
}

K kscalar(const Scalar &s) {
 if(s.isIntegral(false))
  return kj(s.toLong());
 else if(s.isFloatingPoint())
  return kf(s.toDouble());
 AT_ERROR("unexpected scalar type(s), neither integral or floating point, cannot convert");
}

// ------------------------------------------------------------------------------------------
// xlen - 1 if scalar else x->n for lists, no. of table rows or dictionary elements
// mapclass - map class enum to symbol
// kname - string from k data type or object
// ------------------------------------------------------------------------------------------
J xlen(K x) {
 if(x->t < 0 || x->t > 99) return 1;
 else if(x->t < 98)        return x->n;
 else if(x->t == 98)       return xlen(kK(kK(x->k)[1])[0]);
 else                      return xlen(kK(x)[0]);
}

J xlen(K x,J i) {return xind(x,i) ? xlen(kK(x)[i]) : -1;}

S mapclass(Class a) {
 for(auto& m:env().kclass)
  if(a==std::get<1>(m)) return std::get<0>(m);
 AT_ERROR("unrecognized class: ", (I)a);
}

const char* kname(Ktype k) {
 Ktype t=abs(k); bool b=k<0;
 switch(t) {
  case 0: return "general list";
  case 1: return b ? "boolean scalar" : "boolean list";
  case 2: return b ? "guid scalar" : "guid list";
  case 4: return b ? "byte scalar" : "byte list";
  case 5: return b ? "short scalar" : "short list";
  case 6: return b ? "int scalar" : "int list";
  case 7: return b ? "long scalar" : "long list";
  case 8: return b ? "real scalar" : "real list";
  case 9: return b ? "double scalar" : "double list";
  case 10: return b ? "char scalar" : "char list";
  case 11: return b ? "symbol scalar" : "symbol list";
  case 12: return b ? "timestamp scalar" : "timestamp list";
  case 13: return b ? "month scalar" : "month list";
  case 14: return b ? "date scalar" : "date list";
  case 15: return b ? "datetime scalar" : "datetime list";
  case 16: return b ? "timespan scalar" : "timespan list";
  case 17: return b ? "minute scalar" : "minute list";
  case 18: return b ? "second scalar" : "second list";
  case 19: return b ? "time scalar" : "time list";
  case 97: return "nested sym enum";
  case 98: return "table";
  case 99: return "dictionary";
  case 100: return "lambda";
  case 101: return "null/unary primitive";
  case 102: return "operator";
  case 103: return "adverb";
  case 104: return "projection";
  case 105: return "composition";
  case 106: return "f'";
  case 107: return "f/";
  case 108: return "f\\";
  case 109: return "f':";
  case 110: return "f/:";
  case 111: return "f\\:";
  case 112: return "dynamic load fn";
  default:
    if(t>19 && t<77)
     return b ? "enum scalar" : "enum list";
    else if(t>76 && t<97)
     return "map";
    else
     return "value(unrecognized type)";
 }
}

const char* kname(K x) {return xptr(x) ? mapclass(xtag(x)->a) : kname(x->t);}
const char* kname(K x,J i) {return xind(x,i) ? kname(kK(x)[i]) : kname(x);}
 
// ------------------------------------------------------------------------------------------
// ksizeof - given k type, return size of element, e.g. KF -> 8
// maptype - map k data type to/from torch type
// mapattr - make attr enum to symbol
// emap - map sym -> enum (enum used to pick variant member, e.g. torch::kMean)
// ------------------------------------------------------------------------------------------
J ksizeof(Ktype k) {
 switch(k) {
  case KE: return sizeof(E);
  case KF: return sizeof(double);
  case KJ: return sizeof(J);
  case KI: return sizeof(I);
  case KH: return sizeof(H);
  case KC: return sizeof(C);
  case KB:
  case KG: return sizeof(G);
  default: AT_ERROR("no element size for k ",kname(k)); return -1;
 }
}

Ktype maptype(TypeMeta s) {
 for(auto &m:env().dtype)
  if(s==std::get<1>(m)) return std::get<2>(m);
 AT_ERROR("no k data type found for torch type: ",s);
 return 0;
}

TypeMeta maptype(Ktype k) {
 Ktype t=(k<0) ? -k : k;
 for(auto &m:env().ktype)
  if(t==std::get<0>(m)) return std::get<1>(m);
 AT_ERROR("no torch type found for k: ",kname(k));
}

S mapattr(Attr a) {
 for(auto& m:env().attr)
  if(a==std::get<1>(m)) return std::get<0>(m);
 AT_ERROR("unrecognized attribute: ", (I)a);
}

Enum emap(S s) {
 for(const auto &m:env().enums)
  if(std::get<0>(m)==s) return std::get<1>(m);
 return Enum::undefined;
}

// ------------------------------------------------------------------------------------------
// statekey - map from state attribute enumeration to symbol, e.g. State::parms -> `parms
// statefind - search dictionary keys/table colums for symbol matching given enumeration
// statelong - search dict/table for long value
// statesym - given dict/table defining module(s), find symbols for module else null
// statedict - given enumeration, return k dictionary stored at matching key/col else null
// ------------------------------------------------------------------------------------------
S statekey(State e) {
 for(auto &m:env().state)if(e==std::get<1>(m)) return std::get<0>(m);
 AT_ERROR("unrecognized state attribute: ",(I)e);
}

J statefind(State e,K x) {
 if(!xstate(x))
  AT_ERROR("expected dictionary or table describing state, given ",kname(x));
 S s=statekey(e); K k=kK(x->t == 98 ? x->k : x)[0];
 for(J i=0;i<k->n;++i) if(kS(k)[i]==s) return i;
 return -1;
}

static J statelong(State e,bool r,K x,J j) { //e:enum, e.g. State::depth, r:required flag, x:dict/table, j:table row
 J i=statefind(e,x);
 if(i<0) {
  TORCH_CHECK(!r,"unable to find ",statekey(e)," attribute");
  return nj;
 } else if(x->t==99) {
  K v=kK(x)[1];
  if(v->t) {
   TORCH_CHECK(v->t==KJ, statekey(e),": expected long, given ",kname(v));
   return kJ(v)[i];
  } else {
   v=kK(v)[i];
   TORCH_CHECK(v->t==-KJ, statekey(e),": expected long, given ",kname(v));
   return v->j;
  }
 } else if(x->t==98) {
  K v=kK(kK(x->k)[1])[i];
  TORCH_CHECK(v->t==KJ, statekey(e),": expected long, given ",kname(v));
  TORCH_CHECK(j>-1 && j<v->n, statekey(e),"[", j,"] index beyond ",v->n,"-row table");
  return kJ(v)[j];
 } else {
  AT_ERROR("expecting state dictionary or table, given ",kname(x));
 }
}

static S statesym(State e, bool r,K x,J j) { //e:enum, e.g. State::module, r:required, x:dict/table, j:table row
 J i=statefind(e,x);
 if(i<0) {
  TORCH_CHECK(!r,"unable to find `",statekey(e)," attribute");
  return nullptr;
 } else if(x->t==99) {
  K v=kK(x)[1];
  if(v->t) {
   TORCH_CHECK(v->t==KS, statekey(e),": expected symbol, given ",kname(v));
   return kS(v)[i];
  } else {
   v=kK(v)[i];
   TORCH_CHECK(v->t==-KS, statekey(e),": expected symbol, given ",kname(v));
   return v->s;
  }
 } else if(x->t==98) {
  K v=kK(kK(x->k)[1])[i];
  TORCH_CHECK(v->t==KS, statekey(e),": expected symbol, given ",kname(v));
  TORCH_CHECK(j>-1 && j<v->n, statekey(e),"[", j,"] index beyond ",v->n,"-row table");
  return kS(v)[j];
 } else {
  AT_ERROR("expecting state dictionary or table, given ",kname(x));
 }
}

static K statedict(State e,K x,J j) {  // e:enum, e.g. State::options, x:dict/table, j:row (if table)
 J i=statefind(e,x);
 if(i<0) return nullptr;
 K v=x->t == 98 ? kK(kK(x->k)[1])[i] : kK(x)[1];
 if(x->t == 99) j=i;
 TORCH_CHECK(!v->t, statekey(e),": expected dictionary, given ",kname(v));
 TORCH_CHECK(-1<j && j<v->n, statekey(e),"[",j,"] index beyond ",v->n,"-row table");
 v=kK(v)[j];
 TORCH_CHECK(v->t==99, statekey(e),": expected dictionary, given ",kname(v));
 return v;
}

// --------------------------------------------
// convenience functions to return state value
// --------------------------------------------
J statedepth(K x,J j)   {return statelong(State::depth,true,x,j);}
S statemodule(K x,J j)  {return statesym(State::module,true,x,j);}
S statename(K x,J j)    {return statesym(State::name,false,x,j);}
K stateoptions(K x,J j) {return statedict(State::options,x,j);}
K stateparms(K x,J j)   {return statedict(State::parms,x,j);}
K statebuffers(K x,J j) {return statedict(State::buffers,x,j);}

// --------------------------------------------------------------------------------------
// xnull  - true if null, i.e. (::)
// xempty - true if null or empty K list without type, i.e. :: or ()
// atype  - find base type of array by continually looking at 1st element
// xmixed - true if up to m elements of k value has mixed types/lengths
// xsym - if arg is k symbol, return true and set sym, else false
// xsyms - if sym scalar or non-empty sym list, set 1st sym and return true
// xdev  - check sym for map to list of known devices, `cpu`cuda`cuda:0`cuda:1..
// xint64 - check for long scalar/list element and convert to int64_t
// xlong - check for long scalar/list, set value(s) and return true else false
// xdouble - check for scalar double from k, set value and return true, false else
// xdict - return true if k value is a dictionary
// xstate - check for dictionary/table defining module state
// xsize - check for long(s)/double(s), set array ref/expanding array used for sizing
// --------------------------------------------------------------------------------------
bool xnull(K x) {return x->t==101 && x->g==0;}
bool xnull(K x,J i) {return xind(x,i) && xnull(kK(x)[i]);}
bool xempty(K x) {return xnull(x) ? true : (x->t ? false : x->n==0);}
bool xempty(K x,J i) {return xind(x,i) && xempty(kK(x)[i]);}

static Ktype atype(K x) {return (x->t || !x->n) ? x->t : atype(kK(x)[0]);}

bool xmixed(K x,J m) {      // check up to m elements of k value for mixed types/lengths
 Ktype t; J i,n;
 if(!x->t)                                              // if general list
  if(x->n > 1) {                                        // with more than 1 element
   t=atype(kK(x)[0]);                                   // 1st base type encountered
   if(t>19) return true;                                // enums,maps,etc.
   n=t<0 ? 1 : kK(x)[0]->n;                             // 1st size
   if(m>x->n) m=x->n;                                   // check up to m elements
   for(i=1;i<m;++i)
    if(t != atype(kK(x)[i])) return true;               // different data type or scalar vs list
    else if(n != (t<0 ? 1 : kK(x)[i]->n)) return true;  // different length
  }
 return false;
}

bool xsym(K x) {return x->t==-KS;}
bool xsym(K x,J i) {return xind(x,i,KS) || (xind(x,i) && xsym(kK(x)[i]));}
bool xsym(K x,S &s) {return (x->t==-KS) ? s=x->s,true : false;}

bool xsym(K x,J i,S &s) {
 if(xind(x,i,KS))
  return s=kS(x)[i], true;
 else
  return xind(x,i) && xsym(kK(x)[i],s);
}

bool xsyms(K x,S &s) {
 if(xsym(x,s)) return true;
 else if(x->t == KS && x->n) return s=kS(x)[0],true;
 else return false;
}

bool xdev(K x,torch::Device &d) {
 if(x->t==-KS) {
  for(auto &m:env().device)
   if(x->s==std::get<0>(m)) return d=std::get<1>(m),true;
 }
 return false;
}

bool xdev(K x,J i,torch::Device &d) {return xind(x,i) && xdev(kK(x)[i],d);}

bool xint64(K x,int64_t &j) {return (x->t == -KJ) ? j=x->j,true : false;}  //convert J -> int64_t
bool xint64(K x,J i,int64_t &j) {return xind(x,i) && xint64(kK(x)[i],j);}  //mac doesn't differentiate, linux does

bool xlong(K x,J &j) {return (x->t == -KJ) ? j=x->j,true : false;}       //check k scalar
bool xlong(K x,J i,J &j) {return xind(x,i) && xlong(kK(x)[i],j);}        //check k list element

bool xlong(K x,J &n,J *&v){                                        //check for k list of longs
 if(x->t == KJ){          n=x->n; v=kJ(x); return true;            //list of long ints
 } else if(x->t == -KJ){  n=1;    v=&x->j; return true;            //scalar long ok too
 } else if(x->t == 0 && x->n == 0) { n=0;  return true;            //empty,no type also ok
 } else { return false;
 }
}

bool xlong(K x,J i,J &n, J *&v) {return xind(x,i) && xlong(kK(x)[i],n,v);}  // check element of k list

bool xdouble(K x,double &f) {return (x->t == -KF) ? f=x->f,true : false;}    //check k scalar
bool xdouble(K x,J i,double &f) {return xind(x,i) && xdouble(kK(x)[i],f);}   //check k list element

bool xdouble(K x,J &n,double *&v){                                 //check for k list of doubles
 if(x->t == KF){          n=x->n; v=kF(x); return true;            //list of doubles
 } else if(x->t == -KF){  n=1;    v=&x->f; return true;            //scalar double ok too
 } else if(x->t == 0 && x->n == 0) { n=0;  return true;            //empty,no type also ok
 } else { return false;
 }
}
bool xdouble(K x,J i,J &n,double *&v) {return xind(x,i) && xdouble(kK(x)[i],n,v);}  // check element of k list

bool xdict(K x) {return x->t==99 && (kK(x)[0]->t==KS || (kK(x)[0]->t==0 && kK(x)[0]->n==0));}
bool xdict(K x,J i) {return xind(x,i) && xdict(kK(x)[i]);}

bool xstate(K x) {return xdict(x) || x->t==98;}
bool xstate(K x,J i) {return xind(x,i) && xstate(kK(x)[i]);}

// retrieve long integers from x -> IntArrayRef (linux clang/gcc require int64_t* from J*)
bool xsize(K x,IntArrayRef &s) {J n,*v; return (xlong(x,n,v)) ? s=IntArrayRef((int64_t*)v,n),true : false;}
bool xsize(K x,J i,IntArrayRef &s) {return xind(x,i) && xsize(kK(x)[i],s);}  //check element of k list

// retrieve long integers/doubles from x -> ExpandingArray ptr of longs/floats
bool xsize(K x,J d,int64_t *a) {
 bool b=false;
 if((b=x->t == -KJ)) {
   for(J i=0;i<d;++i) a[i]=x->j;
 } else if(x->t == KJ) {
  if((b=d == x->n))
   for(J i=0;i<d;++i) a[i]=kJ(x)[i];
  else
   AT_ERROR(d,"-element list of long integers expected, ",x->n," supplied");
 }
 return b;
}

bool xsize(K x,J d,double *a) {
 bool b=false; 
 if((b=x->t == -KF)) {
  for(J i=0;i<d;++i) a[i]=x->f;
 } else if(x->t == KF) {
  if((b=d == x->n))
   for(J i=0;i<d;++i) a[i]=kF(x)[i];
  else
   AT_ERROR(d,"-element list of doubles expected, ",x->n," supplied");
 }
 return b;
}

bool xsize(K x,J i,J d,int64_t *a) {return xind(x,i) && xsize(kK(x)[i],d,a);}
bool xsize(K x,J i,J d,double  *a) {return xind(x,i) && xsize(kK(x)[i],d,a);}

// ------------------------------------------------------------------------------------------------------
// xten - check arg(s) for allocated ptr to tensor: set tensor & return true if found, else false
//      - 2nd form, return tensor pointer if found from k value, else null
// xvec - check arg(s) for allocated vector of tensors
// ------------------------------------------------------------------------------------------------------
bool xten(K x,Tensor &t) {
 if(auto* a=xtag(x))
  if(a->a==Class::tensor && a->c==Cast::tensor)
   return t=((Kten*)a)->t, true;
 return false;
}

Tensor* xten(K x) {
 if(auto* a=xtag(x))
  if(a->a==Class::tensor && a->c==Cast::tensor)
   return &((Kten*)a)->t;
 return nullptr;
}

bool xten(K x,J i,Tensor& t) {return xind(x,i) && xten(kK(x)[i],t);}
Tensor* xten(K x,J i) {return xind(x,i) ? xten(kK(x)[i]) : nullptr;}

TensorVector* xvec(K x) {
 if(auto* a=xtag(x))
  if(a->a==Class::vector && a->c==Cast::tensor)
   return &((Kvec*)a)->v;
 return nullptr;
}

TensorVector* xvec(K x,J i) {return xind(x,i) ? xvec(kK(x)[i]) : nullptr;}

// ------------------------------------------------------------------------------------------------------
// xtenpair - check arg(s) for a pair of allocated tensor ptrs: if found, set & return true, else false
// xten3 - check arg(s) for a triplet of allocated tensors
// xtenarg - check arg(s) for a list of allocated tensors, or list of input arrays or mix of both
// ------------------------------------------------------------------------------------------------------
bool xtenpair(K x,Tensor& y,Tensor& z) {return xten(x,0,y) && xten(x,1,z);}
bool xtenpair(K x,J i,Tensor& y,Tensor& z) {return xind(x,i) && xtenpair(kK(x)[i],y,z);}
bool xten3(K x,Tensor& t1,Tensor& t2,Tensor& t3) {return xten(x,0,t1) && xten(x,1,t2) && xten(x,2,t3);}
bool xten3(K x,J i,Tensor& t1,Tensor& t2,Tensor& t3) {return xind(x,i) && xten3(kK(x)[i],t1,t2,t3);}
 
bool xtenarg(K x,J i,Tensor& a,Tensor &b) {
 bool p;
 p=xten(x,i,a)   ? true : (a=kput(x,i),false);
 p=xten(x,i+1,b) ? true : (b=kput(x,i+1),p);
 return p;
}

bool xtenarg(K x,J i,Tensor& a,Tensor &b,Tensor &c) {
 bool p;
 p=xten(x,i,a)   ? true : (a=kput(x,i),false);
 p=xten(x,i+1,b) ? true : (b=kput(x,i+1),p);
 p=xten(x,i+2,c) ? true : (c=kput(x,i+2),p);
 return p;
}

bool xtenarg(K x,Tensor& a,Tensor &b)           {return xtenarg(x,0,a,b);}
bool xtenarg(K x,Tensor& a,Tensor &b,Tensor &c) {return xtenarg(x,0,a,b,c);}
 
// ------------------------------------------------------------------------------------------------------
// xmodule - check arg(s) for allocated module pointer
// xlayer - check arg(s) for allocated layer pointer
// xloss - check arg(s) for allocated loss module
// xoptim - check arg(s) for allocated optimizer pointer
// xmodel - check arg(s) for allocated model pointer (module, loss & optimizer)
// ------------------------------------------------------------------------------------------------------
Kmodule* xmodule(K x) {auto* g=xtag(x); return (g && g->a==Class::module) ? (Kmodule*)g : nullptr;}
Kmodule* xmodule(K x,J i) {return xind(x,i) ? xmodule(kK(x)[i]) : nullptr;}
Klayer* xlayer(K x) {auto* g=xtag(x); return (g && g->a==Class::layer) ? (Klayer*)g : nullptr;}
Klayer* xlayer(K x,J i) {return xind(x,i) ? xlayer(kK(x)[i]) : nullptr;}

Kmodule* xloss(K x) {auto* g=xtag(x); return (g && g->a==Class::loss) ? (Kmodule*)g : nullptr;}
Kmodule* xloss(K x,J i) {return xind(x,i) ? xloss(kK(x)[i]) : nullptr;}

Kopt* xoptim(K x) {auto* g=xtag(x); return (g && g->a==Class::optimizer) ? (Kopt*)g : nullptr;}
Kopt* xoptim(K x,J i) {return xind(x,i) ? xoptim(kK(x)[i]) : nullptr;}

Kmodel* xmodel(K x) {auto* g=xtag(x); return (g && g->a==Class::model) ? (Kmodel*)g : nullptr;}
Kmodel* xmodel(K x,J i) {return xind(x,i) ? xmodel(kK(x)[i]) : nullptr;}

// ------------------------------------------------------------------------------------------------------
// xnum - check for double or long int k scalar, set double & return true, else false
// xnum - check for number(float,double,long,int,short), set torch scalar & return true, else false
// xnumn - similar to xnum, but with optional scalars which remain unset if null scalar from k
// xnumt - similar to xnum, but also attempts to convert tensor to scalar
// xnumlist - take single value from k numeric list -> torch scalar
// xbyte - convert k bool,char,byte -> torch scalar
// xscalar - convert k number or byte -> torch scalar
// ------------------------------------------------------------------------------------------------------
bool xnum(K x,double &f) {
 switch(x->t) {
  case -KF: return f=x->f,true;
  case -KJ: return f=x->j,true;
  default: return false;
 }
}
bool xnum(K x,J i,double &f) {
 if(xind(x,i,KF))
  return f=kF(x)[i],true;
 else if(xind(x,i,KJ))
  return f=kJ(x)[i],true;
 else
  return xind(x,i) && xnum(kK(x)[i],f);
}

bool xnum(K x,Scalar& s) {
 switch(x->t) {
  case -KF: return s=x->f, true;
  case -KE: return s=x->e, true;
  case -KJ: return s=(int64_t)x->j, true;
  case -KI: return s=x->i, true;
  case -KH: return s=x->h, true;
  default: return false;
 }
}
bool xnum(K x,J i,Scalar& s) {return xind(x,i) && xnum(kK(x)[i],s);}

bool xnumn(K x,c10::optional<Scalar>& s) {
 switch(x->t) {
  case -KF: if(x->f==x->f) s=x->f; return true;
  case -KE: if(x->e==x->e) s=x->e; return true;
  case -KJ: if(x->j!=nj) s=(int64_t)x->j; return true;
  case -KI: if(x->i!=ni) s=x->i; return true;
  case -KH: if(x->h!=nh) s=x->h; return true;
  default: return false;
 }
}
bool xnumn(K x,J i,c10::optional<Scalar>& s) {return xind(x,i) && xnumn(kK(x)[i],s);}

bool xnumt(K x,Scalar& s) {
 Tensor t;
 if(xnum(x,s))      return true;
 else if(xten(x,t)) return s=t.item(), true;
 else               return false;
}

bool xnumt(K x,J i,Scalar& s) {return xind(x,i) && xnumt(kK(x)[i],s);}

bool xnumlist(K x,J i,Scalar &a) {
 switch(x->t) {
  case KF: return a=kF(x)[i], true;
  case KE: return a=kE(x)[i], true;
  case KJ: return a=(int64_t)kJ(x)[i], true;
  case KI: return a=kI(x)[i], true;
  case KH: return a=kH(x)[i], true;
  case KB:
  case KC: return a=kG(x)[i], true;
  default: return false;
 }
}

bool xbyte(K x,Scalar &s) { return (x->t==-KB || x->t==-KC || xt==-KG) ? s=x->g,true : false;}
bool xbyte(K x,J i,Scalar &s) {return xind(x,i) && xbyte(kK(x)[i],s);}

bool xscalar(K x,Scalar &s) { return xnum(x,s) || xbyte(x,s);}
bool xscalar(K x,J i,Scalar &s) {return xind(x,i) && xscalar(kK(x)[i],s);}

// ------------------------------------------------------------------------------------------------------
// xbool - if value is boolean, set value and return true, else false
// mtype - match sym to/from TypeMeta(newer datatype from Caffe2)
// stype = match sym to/from ScalarType(older ATen datatypes)
// xtype - symbol to scalar type or type meta, return true if scalar type/type meta set, else false
// xopt - sym(s) -> tensor options, return true if ok, false if not sym(s) or error if unknown sym
// xto - device and datatype sym(s) -> tensor options, return true if ok, false if not sym(s)
// xmode - check if sym, if matches a known tensor creation mode, set mode and return true else false
// xbacksym - check if sym, if matches back prop graph setting, set retain/create graph flags else false
// ------------------------------------------------------------------------------------------------------
bool xbool(K x,bool &b) {return (x->t == -KB) ? b=x->g,true : false;}
bool xbool(K x,J i,bool &b) {return xind(x,i) && xbool(kK(x)[i],b);}

TypeMeta mtype(S s) {
  for(auto &m:env().dtype) if(s==std::get<0>(m)) return std::get<1>(m);
  AT_ERROR("unrecognized data type: ",s);
}

S mtype(TypeMeta t) {
  for(auto &m:env().dtype) if(t==std::get<1>(m)) return std::get<0>(m);
  AT_ERROR("unrecognized data type: ",t);
}

ScalarType stype(S s) {return torch::typeMetaToScalarType(mtype(s));}
S stype(ScalarType t) {return mtype(torch::scalarTypeToTypeMeta(t));}
S stype(c10::optional<ScalarType> t) {return mtype(torch::scalarTypeToTypeMeta(t ? *t : ScalarType::Undefined));}

bool xtype(K x,ScalarType &s)                {if(x->t == -KS) return s=stype(x->s), true; return false;}
bool xtype(K x,c10::optional<ScalarType> &s) {if(x->t == -KS) return s=stype(x->s), true; return false;}
bool xtype(K x,TypeMeta   &t) {if(x->t == -KS) return t=mtype(x->s), true; return false;}

bool xtype(K x,J i,ScalarType &s)                {return xind(x,i) && xtype(kK(x)[i],s);}
bool xtype(K x,J i,c10::optional<ScalarType> &s) {return xind(x,i) && xtype(kK(x)[i],s);}
bool xtype(K x,J i, TypeMeta &t) {return xind(x,i) && xtype(kK(x)[i],t);}

bool xopt(S s,TensorOptions &o) {
 auto &e=env();
 for(auto &m:e.device)
  if(s == std::get<0>(m)) return o=o.device(std::get<1>(m)), true;
 for(auto &m:e.dtype)
  if(s == std::get<0>(m)) return o=o.dtype(std::get<1>(m)), true;
 for(auto &m:e.layout)
  if(s == std::get<0>(m)) return o=o.layout(std::get<1>(m)), true;
 for(auto &m:e.gradient)
  if(s == std::get<0>(m)) return o=o.requires_grad(std::get<1>(m)), true;
 return false;
}

bool xopt(K x,TensorOptions &o) {
 if (x->t == -KS || x->t == KS) {
  bool a=x->t < 0; I i,n=a ? 1 : x->n;
  for(i=0; i<n; ++i) {
   S s=a ? x->s : kS(x)[i];
   if (!xopt(s,o))
    AT_ERROR("unrecognized tensor option: `", s);
  }
  return true;
 } else {
  return false;
 }
}

bool xopt(K x,J i,TensorOptions &o) { return !x->t && -1<x->n && i<x->n && xopt(kK(x)[i],o);}

bool xto(S s,TensorOptions &o) {
 for(auto &m:env().device)
  if(s == std::get<0>(m)) return o=o.device(std::get<1>(m)), true;
 for(auto &m:env().dtype)
  if(s == std::get<0>(m)) return o=o.dtype(std::get<1>(m)), true;
 return false;
}

bool xto(K x,TensorOptions &o) { 
 if (x->t == -KS || x->t == KS) {
  bool a=x->t < 0; I i,n=a ? 1 : x->n;
  for(i=0; i<n; ++i) {
   S s=a ? x->s : kS(x)[i];
   if (!xto(s,o))
    AT_ERROR("unrecognized option: `",s,", expecting valid device and/or datatype, e.g. `cpu or `cuda`float");
  }
  return true;
 } else {
  return false;
 }
}

bool xto(K x,J i,TensorOptions &o) {return xind(x,i) ? xto(kK(x)[i],o) : false;}

bool xmode(K x,S &s,Tensormode &m) {
 if(x->t == -KS) {
  for(auto &v:env().tensormode)
   if(x->s == std::get<0>(v)) return s=x->s,m=std::get<1>(v), true;
  AT_ERROR("unrecognized tensor creation mode: ",x->s);
 }
 return false;
}

bool xmode(K x,J i,S &s,Tensormode &m) {return xind(x,i) && xmode(kK(x)[i],s,m);}

bool xbacksym(K x,bool& a,bool& b) {
 if(x->t == -KS) {
  for(auto &s:env().backsym)
   if(x->s == std::get<0>(s)) return a=std::get<1>(s),b=std::get<2>(s), true;
  AT_ERROR("unrecognized setting for backprop: ",x->s,", expecting one of: free,retain,create or createfree");
 }
 return false;
}

bool xbacksym(K x,J i,bool& a,bool& b) {return xind(x,i) && xbacksym(kK(x)[i],a,b);}

// ------------------------------------------------------------------------------------------
// xpairs - initialize a set of name-value pairs given as an argument from k
// xpair - evaluate the next name-value pair, set sym,numeric,list or general value
// xargc - return count of args to process given arg(s), offset, pairs structure to initiate
// xnone - return true if, given arg list and offset, no meaningful arg supplied
// ------------------------------------------------------------------------------------------
bool xpairs(K x,Pairs& p) {   // initialize Pairs structure from k value
 p.a=0, p.i=0, p.n=0;      // sets a: 1-dict,2-pairs,3-list,4-syms
 if(x->t==99) {
  K y=kK(x)[0];
  if(y->t==KS || !(y->t || y->n))
   p.a=1, p.n=y->n;
  else
   AT_ERROR("unexpected name,value dictionary with ",kname(kK(x)[0]->t)," as keys");
 } else if(x->t==KS) {
  if(x->n%2==0)
   p.a=4, p.n=x->n/2;
  else
   AT_ERROR("uneven no. of symbols for name,value pairs: ",x->n);
 } else if(!x->t) {
  if(!x->n) {                      // empty list
   p.a=2, p.n=0;
  } else if(kK(x)[0]->t==-KS) {    // list of (sym1;val1;sym2;val2..)
   if(x->n%2==0)
    p.a=3, p.n=x->n/2;
   else
    AT_ERROR("uneven no. of elements for name,value pairs in list: ",x->n);
  } else {                         // assume list of pairs if symbol in first pair
   K y=kK(x)[0];
   if(y->n==2 && (y->t==KS || (!y->t && kK(y)[0]->t==-KS)))
    p.a=2, p.n=x->n;
  }
 }
 return p.a ? (p.x=x,true) : false;
}

bool xpairs(K x,J i,Pairs& p) {return xind(x,i) && xpairs(kK(x)[i],p);}

static void xpair(Pairs& p,K x,J i) {
 if(x->t<0) {
  switch(x->t) {
   case -KS: p.s=x->s; p.t=-KS; break;
   case -KB: p.b=x->g; p.t=-KB; break;
   case -KH: p.j=x->h; p.t=-KJ; break;
   case -KI: p.j=x->i; p.t=-KJ; break;
   case -KJ: p.j=x->j; p.t=-KJ; break;
   case -KE: p.f=x->e; p.t=-KF; break;
   case -KF: p.f=x->f; p.t=-KF; break;
   default: AT_ERROR("name-value pairs not implemented for ",kname(x->t)," value"); break;
  }
 } else if (i>-1) {
  if(i>=x->n)
   AT_ERROR("name,value index[",i,"] invalid for ",kname(x->t)," with ",x->n," elements");
  switch(x->t) {
   case 0:  xpair(p,kK(x)[i],-1); break;
   case KS: p.s=kS(x)[i]; p.t=-KS; break;
   case KB: p.b=kG(x)[i]; p.t=-KB; break;
   case KH: p.j=kH(x)[i]; p.t=-KJ; break;
   case KI: p.j=kI(x)[i]; p.t=-KJ; break;
   case KJ: p.j=kJ(x)[i]; p.t=-KJ; break;
   case KE: p.f=kE(x)[i]; p.t=-KF; break;
   case KF: p.f=kF(x)[i]; p.t=-KF; break;
   default: AT_ERROR("name-value pairs not implemented for ",kname(x->t)," value"); break;
  }
 } else {
  p.v=x; p.t=x->t;
 }
}

bool xpair(Pairs& p) {
 if(p.i<0 || p.i>=p.n) return false;
 I i=p.i; p.k=nullptr; p.v=nullptr; K y;
 switch(p.a) {   
  case 1:  //dictionary
   p.k=kS(kK(p.x)[0])[i]; xpair(p,kK(p.x)[1],i); break;
  case 2:  //list of name-value pairs
   y=kK(p.x)[i];
   if(xlen(y)!= 2) {
    AT_ERROR("name,value pair[",i,"] has ",xlen(y)," elements (expected 2)");
   } else if(y->t==KS) {
    p.k=kS(y)[0]; xpair(p,y,1);
   } else if(!y->t && kK(y)[0]->t==-KS) {
    p.k=kK(y)[0]->s; xpair(p,kK(y)[1],-1);
   } else {
    AT_ERROR("name,value pair[",i,"] has no name symbol");
   }
   break;
  case 3:  //list of name,value,name,value..
   i*=2; y=kK(p.x)[i];
   if(y->t==-KS) {
    p.k=y->s; xpair(p,kK(p.x)[i+1],-1);
   } else {
    AT_ERROR("unrecognized name for pair, element[",i,"], expected symbol, received: ",kname(y->t));
   }
   break;
  case 4:  // list of symbols
    i*=2; p.k=kS(p.x)[i]; xpair(p,p.x,i+1); break;
  default: AT_ERROR("unrecognized name-value argument"); break;
 }
 return p.i++, true;
}

J xargc(K x,J i,Pairs& p) { // x:arg(s), i:offset, -1 if not applicable, p:pairs to initiate
 if(!x) {
  return 0;
 } else if(xdict(x)) {
  return xpairs(x,p), 0;             // dictionary of options, no other args to process
 } else if(x->t<0 || x->t>97) {
  return i>1 ? 0 : (i<0 ? 1 : 1-i);  // scalar arg, or table or different type of dictionary
 } else if(!x->n) {
  return 0;                          // return 0 for any empty list
 } else if(!(-1<i && i<=x->n)) {
  AT_ERROR("invalid offset: ",i,", for ",kname(x->t)," of length ",x->n);
 } else {
  return x->n-i-(x->n>i ? xpairs(x,x->n-1,p) : false);  // subtract name-value pairs from regular args to process
 }
}

bool xnone(K x,J i) {Pairs p; return !(xargc(x,i,p) || p.n);}

// ------------------------------------------------------------------------------------------
// perr - signal error in type of value given w'name-value pair
// plen  - signal length mismatch of input list if non-negative length supplied
// psym - check for symbol value in name/value pair, return symbol, else error
// ptype - check for symbol value that matches defined data type, e.g. `long`float`double
// pempty - return true if empty array value supplied for current name-value pair
// pbool - check for integral scalar with value 1 or 0, return true/false
// plong - check for integral scalar, return long int
// pdouble - check if numeric scalar, return double
// pnum - check for long/double, set torch scalar
// psize - check if integral scalar or list, set IntArrayRef or ExpandingArray, else error
// pten - attempt to define a tensor from provided scalar or array
// ------------------------------------------------------------------------------------------
void perr(const Pairs& p,const char* s) {AT_ERROR("option: ",p.k," is a ",kname(p.t),", expected a ",s);}

static void plen(const Pairs& p,J n,J m) {
 if(n==0 && (p.t<0 || m)) {
   AT_ERROR("option: ",p.k," requires zero elements, but single scalar value supplied");
 } else if(n>0 && (p.t>=0 && m!=n)) {
  AT_ERROR("option: ",p.k," requires ",n," elements, but ",m," supplied");
 }
}

S psym(const Pairs& p) {if(p.t!=-KS) perr(p,"symbol"); return p.s;}
ScalarType ptype(const Pairs& p) {if(p.t!=-KS) perr(p,"symbol"); return torch::typeMetaToScalarType(mtype(p.s));}
bool pempty(const Pairs& p) {return p.t>=0 && p.v && !p.v->n;}
bool pbool(const Pairs& p) {if(p.t!=-KB) perr(p,"boolean"); return p.b;}
J plong(const Pairs& p) {if(p.t!=-KJ) perr(p,"long integer"); return p.j;}

double pdouble(const Pairs& p) {
 if(!(p.t==-KJ || p.t==-KF)) perr(p,"float, double or integer scalar");
 return p.t==-KJ ? p.j : p.f;
}

void pnum(const Pairs& p,torch::Scalar &s) {
 switch(p.t){
  case -KJ: s=(int64_t)p.j; break;
  case -KF: s=p.f; break;
  default: perr(p,"number"); break;
 }
}

void psize(const Pairs& p,IntArrayRef &s,J n) {
 if(p.t==-KJ)
  s=IntArrayRef((int64_t*)&p.j,1);  // recast for linux clang/gcc to go from J* -> int64_t*
 else if(!(p.t==KJ && xsize(p.v,s)))
  perr(p,"a long integer scalar or list");
 plen(p,n,s.size());
}

void psize(const Pairs& p,J d,int64_t *a) {
 if(p.t == -KJ) {
   for(J i=0;i<d;++i) a[i]=p.j;
 } else if(p.t == KJ) {
  if(d == xlen(p.v))
   for(J i=0;i<d;++i) a[i]=kJ(p.v)[i];
  else
   plen(p,d,xlen(p.v));
 } else {
  perr(p,"long integer scalar or list");
 }
}

void psize(const Pairs& p,J d,double *a) {
 if(p.t == -KF) {
   for(J i=0;i<d;++i) a[i]=p.f;
 } else if(p.t == KF) {
  if(d == xlen(p.v))
   for(J i=0;i<d;++i) a[i]=kF(p.v)[i];
  else
   plen(p,d,xlen(p.v));
 } else {
  perr(p,"double precision scalar or list");
 }
}

void pten(const Pairs& p,Tensor &t) {
 switch(p.t) {
  case 0: if(!(xten(p.v,t))) t=kput(p.v); break;
  case -KB: t=torch::full({},Scalar(p.b),maptype(KB)); break;
  case -KJ: t=torch::full({},Scalar((int64_t)p.j),maptype(KJ)); break;
  case -KF: t=torch::full({},Scalar(p.f),maptype(KF)); break;
  case KB:
  case KH:
  case KI:
  case KJ:
  case KE:
  case KF: t=kput(p.v); break;
  default: perr(p,"numeric scalar/array or previously allocated tensor pointer");
 }
}

// ----------------------------------------------------------------------
// kstring - return string representation of k value
// kout - output k value via "0N!"
// kcast - given data type and array, cast and return, i.e. 1h$x
// kbool - cast k value to boolean
// kdict - tensor dictionary to k dictionary of names -> tensor values
// ----------------------------------------------------------------------
std::string kstring(K x) {
 std::string s;
 K a=k(0,(S)".Q.s1",r1(x),0);
 if(a && a->t==KC) {
  s.assign((S)kC(a),a->n);
  r0(a);
 }
 return s;
}

K kout(K x) {return k(0,(S)"0N!",r1(x),0);}
K kcast(Ktype t,K x) {return k(0,(S)"$",kh(t),r1(x),0);}
K kbool(K x) {return kcast(1,x);}

K kdict(const TensorDict &d) {
 K x=xD(ktn(KS,0),ktn(0,0));
 for(auto &a:d) dictadd(x,a.key().c_str(),kget(a.value()));
 return x;
}

// -----------------------------------------------------------------------------------------
// kfind - given list of symbols, find index of matching string, return -1 if not found
// klist - return k value from count and long/double pointer
// kex - true if given list is one unique value
// kexpand - given element count & data ptr from expanding array return scalar or list
// -----------------------------------------------------------------------------------------
J kfind(K k,const std::string &s) {
 if(k->t != KS) AT_ERROR("unable to look up `",s," in ",kname(k->t),", expecting symbols");
 for(J i=0; i<k->n; ++i) if(!s.compare(kS(k)[i])) return i;
 return -1;
}

K klist(J n,const int64_t *j) {K x=ktn(KJ,n); memcpy(kG(x),j,n*sizeof(int64_t)); return x;}
K klist(J n,const double  *f) {K x=ktn(KF,n); memcpy(kG(x),f,n*sizeof(double));  return x;}
K klist(J n,const c10::optional<int64_t> *j) {
 K x=ktn(KJ,n);
 for(J i=0;i<n;++i) kJ(x)[i] = j[i] ? j[i].value() : nj;
 return x;
}

static bool kex(J n,const c10::optional<int64_t> *e) {
 bool b=n>0;
 for(I i=1;i<n;++i) {
  if((e[i-1].has_value() != e[i].has_value()) ||
     (e[i-1].has_value() && e[i].has_value()  && e[i-1] != e[i]))
   return false;
 }
 return b;
}

template<typename T>static bool kex(J n,const T *e) {
 bool b=n>0; for(I i=1;i<n;++i) if(e[i-1]!=e[i]) return false; return b;
}

K kexpand(J n,const int64_t *e) {return kex<int64_t>(n,e) ? kj(e[0]) : klist(n,e);}
K kexpand(J n,const double  *e) {return kex<double> (n,e) ? kf(e[0]) : klist(n,e);}
K kexpand(J n,const c10::optional<int64_t> *e) {return kex(n,e) ? kj(e[0] ? e[0].value() : nj) : klist(n,e);}

// ---------------------------------------------------------------
// dvt - recursive fn to convert nested tree to (depth;value)
// dv - k api fn convert nested tree -> (depth;value) pairs
// ---------------------------------------------------------------
static K dvt(J d,K x,K z) {
 K y=x->t || !x->n ? x : kK(x)[0];
 if(y->t == -KS)                   
  y=ks(y->s);                      // create a new symbol from given scalar
 else if(y->t == KS && y->n == 1)  
  y=ks(kS(y)[0]);                  // create a new symbol from 1-element symbol vector
 else
  r1(y);                           // else, increment reference count for use directly
 z=jk(&z,knk(2, kj(d), y));        
 if(!x->t)                         // add depth,value pairs from nested list
  for(J i=1;i<x->n;i++)
   z=dvt(d+1,kK(x)[i],z);
 return z;
}

KAPI dv(K x) {
 KTRY
  return dvt(0,x,ktn(0,0));
 KCATCH("dv");
}

// ---------------------------------------------------------------------------------------------
// xdv - return 0 if not recognized as (depth;value) format, -1 for single pair, n-list of pairs
// dve - return x value not recognized as (depth;value)
// dvd - return depth for i'th depth-value arg
// dvv - return i'th value from depth-value scalar/list
// tdv - recursive fn to convert depth-value scalar/list to nested tree representation
// tree - k api function to take depth-value pairs and return nested tree representation
// ---------------------------------------------------------------------------------------------
J xdv(K x) {
 if(x->t) {
  return 0;
 } else if(x->n==2 && kK(x)[0]->t == -KJ) {
  return -1;
 } else {
  for(J i=0;i<x->n;++i)
   if(-1 != xdv(kK(x)[i])) return 0;
  return x->n;
 }
}

static K dve(K x) { 
 if(!x->t && x->n>0 && !xmixed(x,2)) {
  for(J i=0;i<x->n;++i)
   if(!xdv(kK(x)[i])) return kK(x)[i];
 }
 return x;
}

static J dvd(K x,J i) {return kK(i<0 ? x : kK(x)[i])[0]->j;}
static K dvv(K x,J i) {return kK(i<0 ? x : kK(x)[i])[1];}

static K tdv(K x,J i,J j,J n) {
 K v=dvv(x,i), z=ktn(i==j && v->t==-KS ? KS : 0,1);
 if(z->t) kS(z)[0]=v->s;
 else     kK(z)[0]=r1(v);
 J k=i+1;
 for(i=k;i<=j;++i) {
  if(i>k && dvd(x,i)-n==1) {
   z=jk(&z,tdv(x,k,i-1,n+1)); k=i;
  }
  if(i==j)
   z=jk(&z,tdv(x,k,j,n+1));
 }
 return z;
}

KAPI tree(K x) {
 KTRY
  J n=xdv(x);
  TORCH_CHECK(n, "unable to parse (value;depth): ", kstring(dve(x)));
  if(n==-1) return tdv(x, -1,     -1, 0);
  else      return tdv(x,  0, x->n-1, 0);
 KCATCH("tree");
}

// -----------------------------------------------------------------------------------------
// addref - add a new kptr to a shared tensor/module/optimizer, incrementing reference count
// kfree - free allocated object and remove from active pointer set
// Kfree - k-api fn to free a previously allocated object or all allocated objects
// -----------------------------------------------------------------------------------------
KAPI addref(K x) {
 KTRY
  auto *g=xtag(x);
  TORCH_CHECK(g, "addref not implemented for ",kname(x->t));
  switch(g->a) {
   case Class::tensor:     return kten(((Kten*)g)->t);
   case Class::layer:      return klayer(g->c,((Klayer*)g)->m);
   case Class::loss:       return kloss(g->c,((Kmodule*)g)->m);
   case Class::optimizer:  return  kopt(g->c,   ((Kopt*)g)->o);
   default: AT_ERROR("addref not implemented for ",mapclass(g->a));
  }
 KCATCH("addref");
}

bool kfree(K x) {
 if(auto *a=xtag(x))
   return delete a, pointer().erase(kK(x)[0]->j), true;
 else
   return false;
}

bool kfree(K x,J i) {return xind(x,i) ? kfree(kK(x)[i]) : false;}

KAPI Kfree(K x){
 KTRY
  if(xempty(x)) {
   for(auto j:pointer()) delete (Ktag*)j;
   pointer().clear();
  } else {
   TORCH_CHECK(kfree(x), "not a recognized pointer, unable to free");
  }
  return (K)0;
 KCATCH("free");
}

// -----------------------------------------------------------------------------------------
// storsize - storage up to version 1.5 tracked element count and size, now tracks bytes
// -----------------------------------------------------------------------------------------

// -----------------------------------------------------------------------------------------
// objdevice - return tensor device, or first device found if object w'multiple tensors
// objsize - size vector of tensor, else count of parameters/modules
// objnum - number of elements in tensor's underlying storage or sum across tensors
// objbytes - bytes allocated in tensor's storage or sum acress tensors/parms/buffers
// kobj - k api fn returns table of ptr,object,device,dtype,size,number of elements
// -----------------------------------------------------------------------------------------
S objdevice(const Tensor& t) {return tensorsym(t,Attr::device);}
S objdevice(const TensorVector& v,S s) {return v.size() ? objdevice(v[0]) : s;}

static S objdevice(Ktag *x) {
 S s=cs("");
 switch(x->a) {
  case Class::tensor:     return objdevice(((Kten*)x)->t);
  case Class::vector:     return objdevice(((Kvec*)x)->v, s);
  case Class::layer:      return objdevice(layermodule(x).parameters(), s);
  case Class::optimizer:  return objdevice(((Kopt*)x)->o->parameters(), s);
  case Class::model:      return objdevice(layermodule(x).parameters(), s);
  case Class::loss:       return objdevice(((Kmodule*)x)->m.ptr()->buffers(), s);
  default: return s;
 }
}

static J objsize(Kopt *x) {J n=0; for(const auto& a:x->o->param_groups()) n+=a.params().size(); return n;}

static K objsize(Ktag *x) {
 switch(x->a) {
  case Class::tensor:     return tensorsize(((Kten*)x)->t, Attr::size);
  case Class::vector:     return kj(((Kvec*)x)->v.size());
  case Class::layer:
  case Class::model:      return kj(layermodule(x).modules().size());
  case Class::optimizer:  return kj(objsize((Kopt*)x));
  default: return ktn(0,0);
 }
}

static J objnum(const Storage &s) {return s.nbytes() / s.dtype().itemsize();}
static J objnum(const Tensor &t) {return t.is_sparse() ? objnum(t.values()) : objnum(t.storage());}
static J objnum(const TensorVector &v) {J n=0; for(auto& t:v) n+=objnum(t); return n;}
static J objnum(const Module &m) {return objnum(m.parameters());}

static J objnum(Ktag *x) {
 switch(x->a) {
  case Class::tensor:     {auto& a=((Kten*)x)->t; return objnum(a);}
  case Class::vector:     {auto& a=((Kvec*)x)->v; return objnum(a);}
  case Class::layer:
  case Class::model:      return objnum(layermodule(x));
  default: return nj;
 }
}

static J objbytes(const Storage &s) {return s.nbytes();}
static J objbytes(const Tensor &t) {return t.is_sparse() ? objbytes(t.indices())+objbytes(t.values()) : objbytes(t.storage());}
static J objbytes(const TensorVector &v) {J n=0; for(auto& t:v) n+=objbytes(t); return n;}
static J objbytes(const Module &m) {return objbytes(m.parameters()) + objbytes(m.buffers());}

static J objbytes(Ktag *x) {
 switch(x->a) {
  case Class::tensor:     {auto& a=((Kten*)x)->t; return objbytes(a);}
  case Class::vector:     {auto& a=((Kvec*)x)->v; return objbytes(a);}
  case Class::layer:
  case Class::model:      return objbytes(layermodule(x));
  default: return nj;
 }
}


KAPI kobj(K x) {
 KTRY
  TORCH_CHECK(xempty(x), "obj: empty arg expected");
  K k=ktn(KS,7),v=ktn(0,7); auto n=pointer().size(); size_t i=0;
  kS(k)[0]=cs("ptr");      kK(v)[0]=ktn(0,n);
  kS(k)[1]=cs("obj");      kK(v)[1]=ktn(KS,n);
  kS(k)[2]=cs("device");   kK(v)[2]=ktn(KS,n);
  kS(k)[3]=cs("dtype");    kK(v)[3]=ktn(KS,n);
  kS(k)[4]=cs("size");     kK(v)[4]=ktn(0,n);
  kS(k)[5]=cs("elements"); kK(v)[5]=ktn(KJ,n);
  kS(k)[6]=cs("bytes");    kK(v)[6]=ktn(KJ,n);
  for(auto j:pointer()) {
   auto *g=(Ktag*)j;
   kK(kK(v)[0])[i] = knk(1,kj(j));
   kS(kK(v)[1])[i] = mapclass(g->a);
   kS(kK(v)[2])[i] = objdevice(g);
   kS(kK(v)[3])[i] = g->a == Class::tensor ? tensorsym(((Kten*)g)->t, Attr::dtype)  : cs("");
   kK(kK(v)[4])[i] = objsize(g);
   kJ(kK(v)[5])[i] = objnum(g);
   kJ(kK(v)[6])[i] = objbytes(g);
   ++i;
  }
  return xT(xD(k,v));
 KCATCH("obj");
}

// -----------------------------------------------------------------------------------------
// kstate - retrieve module/loss/optimizer state: options, internal buffers & parameters
// to - convert tensor/module device and or data type, e.g. to[tensor;`cuda`float;0b]
// kdetail - return dictionary of attributes of given object and level of detail
// -----------------------------------------------------------------------------------------
KAPI kstate(K x) {
 KTRY
  Ktag *g;
  if(!((g=xtag(x)) || (g=xtag(x,0))))
   AT_ERROR("state expects a pointer to previously allocated module, optimizer or loss function");
  switch(g->a) {
   //case Class::sequential: return mstate(x);
   case Class::loss:       return lossdict(g,x);
   case Class::optimizer:  return optstate(g,x);
   case Class::model:      return modelstate(g,x);
   case Class::tensor:
   case Class::vector:     AT_ERROR("state not defined for ",mapclass(g->a));
   default: return KERR("not a recognized pointer");
  }
 KCATCH("state")
}

KAPI to(K x) {
 KTRY
  bool a=false,b=false; Ktag *g; Tensor t; TensorOptions o;
  if((g=xtag(x,0)) && (xto(x,1,o) || xten(x,1,t)) &&
     (x->n==2 || (xbool(x,2,a) && (x->n==3 || (x->n==4 && xbool(x,3,b)))))) {
   TORCH_CHECK(b ? g->a == Class::tensor : true,
               "4th arg, copy flag, can not be true for ",mapclass(g->a));
   if(t.defined())
    o=o.device(t.device()).dtype(t.dtype());
   switch(g->a) {
    case Class::tensor:     return tento((Kten*)g,o,a,b);
    case Class::vector:     return vecto((Kvec*)g,o,a);
    case Class::layer:      return layerto((Klayer*)g,o,a);
    case Class::loss:       return lossto((Kmodule*)g,o,a);
    default: AT_ERROR("to() not implemented for: ",mapclass(g->a));
   }
  } else {
   AT_ERROR("to: unrecognized arg(s)");
  }
 KCATCH("to");
}

static K kinfo(K x,bool b,const char* e) {
 KTRY
  auto* g=xtag(x);
  TORCH_CHECK(g, e," not implemented for ",kname(x->t));
  switch(g->a) {
   case Class::tensor:     return tensorinfo(((Kten*)g)->t,b);
   default: AT_ERROR(e," not implemented for ",mapclass(g->a));
  }
 KCATCH(e);
}

KAPI info1(K x) {return kinfo(x, false, "info");}
KAPI info2(K x) {return kinfo(x, true,  "detail");}

// -----------------------------------------------------------------------------------------
// zerograd - zero gradients on tensor, vector of tensors, optimizer, sequential or model
// forward - forward calcs on sequential module or model
// kbackward - backward calcs on tensor or model(uses model loss(model output,target) )
// -----------------------------------------------------------------------------------------
KAPI zerograd(K x) {
 KTRY
  auto *g=xtag(x);
  auto f=[](Tensor& t) {if(t.grad().defined()) t.grad().detach().zero_();};
  TORCH_CHECK(g, "zerograd not implemented for ",kname(x->t));
  switch(g->a) {
   case Class::tensor:     f(((Kten*)g)->t); break;
   case Class::vector:     for(auto& t:((Kvec*)g)->v) f(t); break;
   case Class::optimizer:  ((Kopt*)g)->o->zero_grad(); break;
   case Class::model:      ((Kmodel*)g)->o->zero_grad(); break;
   default: AT_ERROR("zerograd not implemented for ",mapclass(g->a));
  }
  return (K)0;
 KCATCH("zero gradients");
}

KAPI forward(K x) {
 KTRY
  Ktag *g;
  TORCH_CHECK((g=xtag(x,0)), "forward expects layer(s) or full model as first arg");
  switch(g->a) {
   case Class::layer:      return layerforward(((Klayer*)g)->m,x);
   case Class::model:      return layerforward(((Kmodel*)g)->m,x);
   default: AT_ERROR("forward not implemented for ",mapclass(g->a));
  }
  return (K)0;
 KCATCH("forward");
}

KAPI kbackward(K x) {
 KTRY
  Ktag *g;
  TORCH_CHECK((g=xtag(x)) || (g=xtag(x,0)), "backward expects a tensor or model as first arg");
  switch(g->a) {
   case Class::tensor: return tensorback(x);
   case Class::model:  return mbackward(x);
   default: AT_ERROR("backward not implemented for ",mapclass(g->a));
  }
  return (K)0;
 KCATCH("backward");
}

// ---------------------------------------------------------------------------------------------
// cudadevices - return number of CUDA devices enabled or available CUDA device symbols
// cudadevice - k interface to set/query current CUDA device, e.g. `cuda:0 
// defaultdevice - return `cuda if any cuda devices available, else `cpu
// ---------------------------------------------------------------------------------------------
KAPI cudadevices(K x) {
 if(xnull(x)) {
  return kj(env().cuda);
 } else if(xempty(x)) {
  K s=ktn(KS,0);
  for(auto& m:env().device) if((std::get<1>(m)).is_cuda()) js(&s,std::get<0>(m));
  return s;
 } else {
  return KERR("cudadevices[] returns count of available GPUs, cudadevices() returns CUDA syms");
 }
}

KAPI cudadevice(K x) {
 KTRY
  TORCH_CHECK(env().cuda, "no CUDA device available");
  torch::Device d(torch::kCUDA);
  auto *g = c10::impl::getDeviceGuardImpl(d.type());
  if(xempty(x)) {
   for(auto &m:env().device)
    if(g->getDevice()==std::get<1>(m)) return ks(std::get<0>(m));
   AT_ERROR("unable to map CUDA device: ",g->getDevice().index()," to symbol");
  } else if(xdev(x,d) && d.is_cuda() && d.has_index()) {
   return g->setDevice(d), K(0);
  } else {
   return KERR("unrecognized CUDA device, expecting cuda with valid device number, e.g. `cuda:0");
  }
 KCATCH("unable to query/set CUDA device")
}

static K defaultdevice() {
 auto d=torch::Device(env().cuda ? torch::DeviceType::CUDA : torch::DeviceType::CPU);
 for(auto& c:env().device)
  if(std::get<1>(c)==d) return ks(std::get<0>(c));
 return KERR("unable to get default device");
}

// ---------------------------------------------------------------------------------------------
// optsym - given tensor options, return underlying device,data type,layout & grad/nograd as sym
// optmap - given tensor options, return dictionary of attribute -> setting
// optkey - symbol keys/cols for tensor option dictionary or table
// optval - symbol vector/lists of option values
// ---------------------------------------------------------------------------------------------
S& optsym(const torch::Device& d) {
 for(auto &m:env().device) if(d==std::get<1>(m)) return std::get<0>(m);
 AT_ERROR("unrecognized device: ",d);
}

S& optsym(const TypeMeta& t) {
 for(auto &m:env().dtype) if(t==std::get<1>(m)) return std::get<0>(m);
 AT_ERROR("unrecognized data type: ",t);
}

S& optsym(const torch::Layout& l) {
 for(auto &m:env().layout) if(l==std::get<1>(m)) return std::get<0>(m);
 AT_ERROR("unrecognized layout: ",l);
}

S& optsym(const bool& g) {
 for(auto &m:env().gradient) if(g==std::get<1>(m)) return std::get<0>(m);
 AT_ERROR("unrecognized gradient setting: ",g);
}

K optkey() {
 K x=ktn(KS,4);
 kS(x)[0]=mapattr(Attr::device);
 kS(x)[1]=mapattr(Attr::dtype);
 kS(x)[2]=mapattr(Attr::layout);
 kS(x)[3]=mapattr(Attr::gradient);
 return x;
}

K optval(const TensorOptions &o,K x,J i) {
 if(x->t==KS) {
  kS(x)[0]=optsym(o.device());
  kS(x)[1]=optsym(o.dtype());
  kS(x)[2]=optsym(o.layout());
  kS(x)[3]=optsym(o.requires_grad());
 } else {
  kS(kK(x)[0])[i]=optsym(o.device());
  kS(kK(x)[1])[i]=optsym(o.dtype());
  kS(kK(x)[2])[i]=optsym(o.layout());
  kS(kK(x)[3])[i]=optsym(o.requires_grad());
 }
 return x;
}

K optmap(const TensorOptions &o) {
 return xD(optkey(),optval(o,ktn(KS,4)));
}

// ---------------------------------------------------------------------------------------------
// kdefault - k interface to query/set default tensor options
// ksetting - list/change configuration settings
// config - print or return strings of pytorch config (CUDA capability, build options, etc.)
// ---------------------------------------------------------------------------------------------
KAPI kdefault(K x) {
 torch::TensorOptions o;
 KTRY
  if(xempty(x)) {
   return optmap(o);
  } else if(xopt(x,o)) {
   TORCH_CHECK(!(o.has_device() || o.has_layout() || o.has_requires_grad()),
               "currently, only default data type can be reset");
   torch::set_default_dtype(o.dtype());
   return(K)0;
  } else {
   return KERR("unrecognized argument for querying/setting default tensor options");
  }
 KCATCH("default");
}

KAPI ksetting(K x) {
 KTRY
  auto &e=env(); auto &c=at::globalContext(); bool b,o=torch::hasOpenMP(); J n; S s;
  if(xempty(x)) {
   K r=xD(ktn(KS,0),ktn(0,0)),*s=&kK(r)[0],*v=&kK(r)[1];
   js(s,cs("mkl"));            jk(v,kb(torch::hasMKL()));
   js(s,cs("openmp"));         jk(v,kb(o));
   js(s,cs("threads"));        jk(v,kj(o ? torch::get_num_threads() : 1));
   js(s,cs("cuda"));           jk(v,kb(torch::cuda::is_available()));
   js(s,cs("magma"));          jk(v,kb(torch::hasMAGMA()));
   js(s,cs("cudnn"));          jk(v,kb(torch::cuda::cudnn_is_available()));
   js(s,cs("cudnnversion"));   jk(v,kj(torch::cuda::cudnn_is_available()
                                    ? at::detail::getCUDAHooks().versionCuDNN() : nj));
   js(s,cs("cudadevices"));    jk(v,kj(e.cuda));
   js(s,cs("benchmark"));      jk(v,kb(c.benchmarkCuDNN()));
   js(s,cs("deterministic"));  jk(v,kb(c.deterministicCuDNN()));
   js(s,cs("stackframe"));     jk(v,kb(e.frame));
   js(s,cs("alloptions"));     jk(v,kb(e.alloptions));
   return r;
  } else if (xsym(x,0,s) && xbool(x,1,b) && x->n==2) {
   if(s==cs("benchmark"))           c.setBenchmarkCuDNN(b);
   else if(s==cs("deterministic"))  c.setDeterministicCuDNN(b);
   else if(s==cs("stackframe"))     e.frame=b;
   else if(s==cs("alloptions"))     e.alloptions=b;
   else                             AT_ERROR("unable to change setting: ",s);
   return(K)0;
  } else if (xsym(x,0,s) && s==cs("threads") && xlong(x,1,n) && x->n==2) {
   if(!o) AT_ERROR("unable to set number of threads, OpenMP not available");
   torch::set_num_threads(n);
   return(K)0;
  } else {
   return KERR("unrecognized arg(s) -- use empty arg to query, use (sym;bool) to set one of `benchmark`deterministic`stackframe`alloptions or (`threads;n) for threads");
  }
 KCATCH("unable to query/change torch settings");
}

KAPI config(K x) {
 KTRY
  auto c1=torch::show_config(),c2=torch::get_parallel_info();
  if(xnull(x)) {
   std::cerr << c1 << "\n";
   std::cerr << c2 << "\n";
   return (K)0;
  } else if(xempty(x)) {
   std::stringstream s1(c1),s2(c2); std::string t; K z=ktn(0,0);
   while(std::getline(s1,t,'\n')) jk(&z,kp((S)t.c_str()));
   while(std::getline(s2,t,'\n')) jk(&z,kp((S)t.c_str()));
   return z;
  } else {
   return KERR("config expects empty argument: config[] prints to stderr, config() returns strings");
  }
 KCATCH("config");
}

// -----------------------------------------------------------------------------------
// deviceseed - query/set seed for given device, return initial seed in use for device
// seedmap - returns map of device sym -> seed
// kseed - k interface to query/set device seed or query/reset seed for all devices
// -----------------------------------------------------------------------------------
J deviceseed(torch::Device &d, bool b=false,J s=0) { // d:device, b:set flag, s:seed to set
 return 0;
 torch::DeviceGuard dg(d);
 // PATCH auto &g=at::globalContext().defaultGenerator(d.is_cuda() ? torch::kCUDA : torch::kCPU); // version 1.5
 auto g=at::globalContext().defaultGenerator(d.is_cuda() ? torch::kCUDA : torch::kCPU);  // version 1.6
 if(b) {
  if(null(s))
   g.seed();
  else
   g.set_current_seed(s);
 }
 return g.current_seed();
}

static K seedmap() {
 auto a=env().device; auto n=a.size(); I i=0; K k=ktn(KS,n),v=ktn(KJ,n);
 for(auto& m:a)
  kS(k)[i]=std::get<0>(m),kJ(v)[i++]=deviceseed(std::get<1>(m));
 return xD(k,v);
}

KAPI kseed(K x) {
 KTRY
  torch::Device d(torch::DeviceType::CPU); J s;
  if(xempty(x)) {                 // if empty, report on seed for all devices
   return seedmap();
  } else if(xlong(x,s)) {         // set single random seed across all devices
   // PATCH if(null(s)) s=at::detail::getNonDeterministicRandom();  // version 1.5
   if(null(s)) s=c10::detail::getNonDeterministicRandom(); // version 1.6
   torch::manual_seed(s);
   return (K)0;
  } else if(xdev(x,d)) {          // query initial random seed for given device
   return kj(deviceseed(d));
  } else if(xdev(x,0,d) && xlong(x,1,s) && x->n==2) {  // set seed for given device
   deviceseed(d,true,s);
   return (K)0;
  } else {
   return KERR("unrecognized arg(s) for seed, expected one of: device, seed or (device;seed)");
  }
 KCATCH("unable to set/retrieve random seed(s)");
}

// -----------------------------------------------------------------------------------------
// query object attributes, e.g. tensor/vector and other object attributes
// -----------------------------------------------------------------------------------------
static K attr(K x,Ktype k,Attr a) {
 KTRY
  auto* g=xtag(x);
  TORCH_CHECK(g, mapattr(a),": unrecognized arg(s) - ",kname(x->t));
  switch(g->a) {
   case Class::tensor:     return tensorattr(((Kten*)g)->t,k,a);
   case Class::vector:     return vectorattr(((Kvec*)g)->v,k,a);
   case Class::layer:      return layerattr(((Klayer*)g)->m,k,a);
   case Class::loss:       return lossattr(((Kmodule*)g)->m,k,a);
   case Class::optimizer:  return optattr(((Kopt*)g)->o,k,a);
   default: AT_ERROR(mapattr(a),": not implemented for ",mapclass(g->a));
  }
 KCATCH("attr");
}

KAPI        dim(K x) {return attr(x, -KJ, Attr::dim);}
KAPI      numel(K x) {return attr(x, -KJ, Attr::numel);}
KAPI     offset(K x) {return attr(x, -KJ, Attr::offset);}
KAPI        ref(K x) {return attr(x, -KJ, Attr::ref);}
KAPI    weakref(K x) {return attr(x, -KJ, Attr::weakref);}
KAPI        ptr(K x) {return attr(x, -KJ, Attr::ptr);}
KAPI    storage(K x) {return attr(x, -KJ, Attr::storage);}

KAPI     device(K x) {return xempty(x) ? defaultdevice() : attr(x, -KS, Attr::device);}
KAPI      dtype(K x) {return attr(x, -KS, Attr::dtype);}
KAPI     layout(K x) {return attr(x, -KS, Attr::layout);}
KAPI   gradient(K x) {return attr(x, -KS, Attr::gradient);}
KAPI     gradfn(K x) {return attr(x, -KS, Attr::gradfn);}

KAPI contiguous(K x) {return attr(x, -KB, Attr::contiguous);}
KAPI       leaf(K x) {return attr(x, -KB, Attr::leaf);}
KAPI     pinned(K x) {return attr(x, -KB, Attr::pinned);}

KAPI       size(K x) {return attr(x,  KJ, Attr::size);}
KAPI     stride(K x) {return attr(x,  KJ, Attr::stride);}

// ----------------------------------------------------------------------------------------------
// filewrite - write zero bytes if file writable, create any necessary dir(s), drop leading colon
//  pt
// png - write PNG file given name & array/tensor
// ----------------------------------------------------------------------------------------------
S filewrite(S s) {k(0,(S)"1:",ks(s),ktn(KB,0),0); return s[0]==':' ? s+1 : s;}

KAPI pt(K x) {
 KTRY
  S s; Tensor t;
  if(xsym(x,s)) {
   torch::load(t,s[0]==':' ? ++s : s);
   return kget(t);
  } else if (xsym(x,0,s) && xten(x,1,t)) {
   torch::save(t,s[0]==':' ? ++s : s);
  }
  return (K)0;
 KCATCH("pt");
}

KAPI png(K x) {
 KTRY
  S s; Tensor t;
  TORCH_CHECK(xsym(x,0,s) && x->n==2, "png: unrecognized arg(s), expects (PNG file name; image array/tensor)");
  if(!xten(x,1,t)) t=kput(x,1);
  auto a=t.to(torch::kCPU,torch::kByte).permute({1,2,0}).flatten(1);
  stbi_write_png(filewrite(s),t.size(2),t.size(1),t.size(0),a.data_ptr(),t.size(0)*t.size(2));
  return ks(s);
 KCATCH("png");
}

// -----------------------------------------------------------------------------------------
// initialize globals: device counts, device sym-int mapping, etc.
// kinit - called when shared library is first opened
// -----------------------------------------------------------------------------------------
Env&  env()  {static Env  e; return e;}
Esym& esym() {static Esym e; return e;}
std::unordered_set<J>& pointer() {static std::unordered_set<J> p; return p;}

static void kinit() __attribute__((constructor));

static void kinit() {
 C c[16]; auto &e=env(); auto &d=e.device;
 e.frame = false;                                                     //no stack frame on error msg
 e.cuda = torch::cuda::device_count();                                //count of available CUDA devices
 d.emplace_back(cs("cpu"),torch::Device(torch::DeviceType::CPU));     //build map from sym->device
 if(e.cuda) {
  d.emplace_back(cs("cuda"),torch::Device(torch::DeviceType::CUDA));  //current CUDA device, `cuda
  for(I i=0; i<e.cuda; ++i) {
   sprintf(c,"cuda:%d",i);                                            //device 0-n, e.g. `cuda:0
   d.emplace_back(ss(c),torch::Device(torch::DeviceType::CUDA,i));
  }
 }
}

// -----------------------------------------------------------------------------------------
// fn - given dictionary, along with name, fn & arg count, adds function to dictionary
// fns - returns K dictionary with function names and code
// -----------------------------------------------------------------------------------------
void fn(K x,const char* s,void *f,I n){dictadd(x,s,dl(f,n));}

KAPI fns(K x){
 x=xD(ktn(KS,0),ktn(0,0));
 fn(x, "dv",          KFN(dv),          1);
 fn(x, "tree",        KFN(tree),        1);
 fn(x, "addref",      KFN(addref),      1);
 fn(x, "free",        KFN(Kfree),       1);
 fn(x, "obj",         KFN(kobj),        1);
 fn(x, "to",          KFN(to),          1);
 fn(x, "info",        KFN(info1),       1);
 fn(x, "detail",      KFN(info2),       1);
 fn(x, "state",       KFN(kstate),      1);
 fn(x, "forward",     KFN(forward),     1);
 fn(x, "zerograd",    KFN(zerograd),    1);
 fn(x, "backward",    KFN(kbackward),   1);
 fn(x, "default",     KFN(kdefault),    1);
 fn(x, "setting",     KFN(ksetting),    1);
 fn(x, "config",      KFN(config),      1);
 fn(x, "cudadevice",  KFN(cudadevice),  1);
 fn(x, "cudadevices", KFN(cudadevices), 1);
 fn(x, "seed",        KFN(kseed),       1);
 fn(x, "png",         KFN(png),         1);

 fn(x, "dim",         KFN(dim),         1);
 fn(x, "numel",       KFN(numel),       1);
 fn(x, "offset",      KFN(offset),      1);
 fn(x, "ptr",         KFN(ptr),         1);
 fn(x, "ref",         KFN(ref),         1);
 fn(x, "storage",     KFN(storage),     1);
 fn(x, "weakref",     KFN(weakref),     1);
 fn(x, "device",      KFN(device),      1);
 fn(x, "dtype",       KFN(dtype),       1);
 fn(x, "gradfn",      KFN(gradfn),      1);
 fn(x, "gradient",    KFN(gradient),    1);
 fn(x, "layout",      KFN(layout),      1);
 fn(x, "contiguous",  KFN(contiguous),  1);
 fn(x, "leaf",        KFN(leaf),        1);
 fn(x, "pinned",      KFN(pinned),      1);
 fn(x, "size",        KFN(size),        1);
 fn(x, "stride",      KFN(stride),      1);

 tensorfn(x);
 mathfn(x);
 nnfn(x);
 lossfn(x);
 optfn(x);
 modelfn(x);
 return x;
}
