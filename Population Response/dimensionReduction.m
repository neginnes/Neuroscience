function MAT = dimensionReduction(data,nDimentsions)
    coeff = pca(data');
    MAT = data'*coeff(:, 1:nDimentsions); 
    MAT = MAT';
end