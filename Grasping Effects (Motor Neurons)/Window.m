function MAT = Window(data,window,row)
    n = floor(length(data(1,:))/window);
    MAT = zeros(n,1);
    for j=1:n
        MAT(j) = mean(data(row,(j-1)*window+1:j*window));
    end
    MAT = MAT';
end