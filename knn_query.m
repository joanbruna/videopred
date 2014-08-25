function knn_query(X, nnmax, timeth)


vl_setup
tree=vl_kdtreebuild(X);

[nnid, ndist] = vl_kdtreequery(tree,X,X, 'NUMNEIGHBORS',nnmax,'MAXCOMPARISONS',250) ;





