function input = edgeHandler(input,mapSize)
    INF = -10e9;
    input(1,:,3) = INF;
    input(mapSize,:,1) = INF;
    input(:,1,2) = INF;
    input(:,mapSize,4) = INF;   
end