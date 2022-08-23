function mapValues = updateValue(mapValues,uniformProbs,actProbs,currentratPos,gamma,reward)
    mapSize = size(mapValues);
    x = currentratPos(1);
    y = currentratPos(2);
    if (reward(currentratPos(1),currentratPos(2))==0)
        sum2D = 0; 
        if (x == 1 && y == 1)
            for i = [1,4]
                for j = [1,4]
                   if(j == 1)
                        nextx = x + 1;
                        nexty = y;
                    elseif(j == 4)
                        nextx = x;
                        nexty = y+1;
                    end
                    sum2D = sum2D + actProbs(x,y,i)*uniformProbs(x,y,j)*mapValues(nextx,nexty);
                end
            end
        elseif (x == 1 && y == mapSize(2))
            for i = [1,2]
                for j = [1,2]
                   if(j == 1)
                        nextx = x + 1;
                        nexty = y;
                    elseif(j == 2)
                        nextx = x;
                        nexty = y-1;
                    end
                    sum2D = sum2D + actProbs(x,y,i)*uniformProbs(x,y,j)*mapValues(nextx,nexty);
                end
            end            
        elseif (x == 1)
            for i = [1,2,4]
                for j = [1,2,4]
                    if(j == 1)
                        nextx = x + 1;
                        nexty = y;
                    elseif(j == 2)
                        nextx = x;
                        nexty = y-1;
                    elseif(j == 4)
                        nextx = x;
                        nexty = y+1;
                    end
                    sum2D = sum2D + actProbs(x,y,i)*uniformProbs(x,y,j)*mapValues(nextx,nexty);
                end
            end  
        elseif (x == mapSize(1) && y == 1)
            for i = [3,4]
                for j = [3,4]
                    if(j == 3)
                        nextx = x-1;
                        nexty = y;
                    elseif(j == 4)
                        nextx = x;
                        nexty = y+1;
                    end
                    sum2D = sum2D + actProbs(x,y,i)*uniformProbs(x,y,j)*mapValues(nextx,nexty);
                end
            end
        elseif(x == mapSize(1) && y == mapSize(2))
            for i = [2,3]
                for j = [2,3]
                    if(j == 2)
                        nextx = x;
                        nexty = y-1;
                    elseif(j == 3)
                        nextx = x-1;
                        nexty = y;
                    end
                    sum2D = sum2D + actProbs(x,y,i)*uniformProbs(x,y,j)*mapValues(nextx,nexty);
                end
            end
        elseif (y == 1)
            for i = [1,3,4]
                for j = [1,3,4]
                    if(j == 1)
                        nextx = x + 1;
                        nexty = y;
                    elseif(j == 3)
                        nextx = x-1;
                        nexty = y;
                    elseif(j == 4)
                        nextx = x;
                        nexty = y+1;
                    end
                    sum2D = sum2D + actProbs(x,y,i)*uniformProbs(x,y,j)*mapValues(nextx,nexty);
                end
            end    
        elseif(x == mapSize(1))
            for i = [2,3,4]
                for j = [2,3,4]
                    if(j == 2)
                        nextx = x;
                        nexty = y-1;
                    elseif(j == 3)
                        nextx = x-1;
                        nexty = y;
                    elseif(j == 4)
                        nextx = x;
                        nexty = y+1;
                    end
                    sum2D = sum2D + actProbs(x,y,i)*uniformProbs(x,y,j)*mapValues(nextx,nexty);
                end
            end
        elseif(y == mapSize(2))
            for i = [1,2,3]
                for j = [1,2,3]
                    if(j == 1)
                        nextx = x + 1;
                        nexty = y;
                    elseif(j == 2)
                        nextx = x;
                        nexty = y-1;
                    elseif(j == 3)
                        nextx = x-1;
                        nexty = y;
                    end
                    sum2D = sum2D + actProbs(x,y,i)*uniformProbs(x,y,j)*mapValues(nextx,nexty);
                end  
            end
        else    
            for i = [1,2,3,4]
                for j = [1,2,3,4]
                    if(j == 1)
                        nextx = x + 1;
                        nexty = y;
                    elseif(j == 2)
                        nextx = x;
                        nexty = y-1;
                    elseif(j == 3)
                        nextx = x-1;
                        nexty = y;
                    elseif(j == 4)
                        nextx = x;
                        nexty = y+1;
                    end
                    sum2D = sum2D + actProbs(x,y,i)*uniformProbs(x,y,j)*mapValues(nextx,nexty);
                end
            end  
        end
        mapValues(x,y) = reward(x,y)+gamma*sum2D;
    else
        mapValues(x,y) = reward(x,y);
    end
end