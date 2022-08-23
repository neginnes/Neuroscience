function mapValues = TDLamdaUPdator(mapValuesBefore,mapValues,visitedLocs,currentPos,lambda)
    delta = mapValues(currentPos(1),currentPos(2)) - mapValuesBefore(currentPos(1),currentPos(2));
    for tprime = 1 : size(visitedLocs,1)
        mapValues(visitedLocs(tprime,1),visitedLocs(tprime,2)) = mapValues(visitedLocs(tprime,1),visitedLocs(tprime,2)) + (lambda^(size(visitedLocs,1)-tprime + 1))*delta;
    end
end