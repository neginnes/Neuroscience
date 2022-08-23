function pathGenerator(mapSize,targetPos,catPos,ratPos,ratPosPre)
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
    hold on
    plot([ratPosPre(1),ratPos(1)],[ratPosPre(2),ratPos(2)],'blue');
    hold on
    scatter(ratPos(1),ratPos(2),'*','blue')
    
end