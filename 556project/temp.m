close all hidden

tempnew = [];
for k = 1:200
  tempnew(k) = random('poiss', 14);
end
%figure(1); plot(tempnew);
tempnew