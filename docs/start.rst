.. _start:

Quick start
===========

::

   q)(`ktorch 2:`fns,1)[]
   dv         | code
   tree       | code
   addref     | code
   free       | code
   obj        | code
   ..

   q){key[x]set'x;}(`ktorch 2:`fns,1)[]
   
   q)tensor t:tensor 1; show obj[]; free[]
   ptr      obj    device dtype size elements bytes
   ------------------------------------------------
   37532000 tensor cpu    long       1        8    
  
   q).nn:(`ktorch 2:`fns,1)[]
   q).nn.tensor t:.nn.tensor 1; show .nn.obj[]; .nn.free[]
   ptr      obj    device dtype size elements bytes
   ------------------------------------------------
   37525424 tensor cpu    long       1        8    
   
