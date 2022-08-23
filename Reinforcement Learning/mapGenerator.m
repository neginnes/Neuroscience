function mapGenerator(mapSize,targetPos,catPos,ratPos)
    set(gcf,'color','w');
    for i = 1 : mapSize + 1
        x = (i-1)*ones(mapSize + 1,1);
        y = 0 : mapSize;
        plot(x,y,'k');
        plot(y,x,'k');
        hold on
    end
    Xtarget = [targetPos(1)-0.5,targetPos(1)-0.5,targetPos(1)+0.5,targetPos(1)+0.5];
    Ytarget = [targetPos(2)-0.5,targetPos(2)+0.5,targetPos(2)+0.5,targetPos(2)-0.5];
    fill(Xtarget,Ytarget,'black');
    hold on
    Xcat = [catPos(1)-0.5,catPos(1)-0.5,catPos(1)+0.5,catPos(1)+0.5];
    Ycat = [catPos(2)-0.5,catPos(2)+0.5,catPos(2)+0.5,catPos(2)-0.5];
    fill(Xcat,Ycat,'red');
    hold on
    Xrat = [ratPos(1)-0.5,ratPos(1)-0.5,ratPos(1)+0.5,ratPos(1)+0.5];
    Yrat = [ratPos(2)-0.5,ratPos(2)+0.5,ratPos(2)+0.5,ratPos(2)-0.5];
    fill(Xrat,Yrat,'yellow');
    hold off
end