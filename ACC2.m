function [ACCIndex lableRef] = ACC2(gnd,lable,C)
%%%%%%%%gnd ��׼���ࣻlable:��õĽ����C ����%%%%%%%%%%%%%
  
 lableRef = bestMap(gnd,lable);

 ACCIndex = sum(lableRef(:) == gnd(:)) / length(gnd);