close all hidden

tempnew = [];
for k = 1:10000
  tempnew(k) = random('poiss', 14);
end
figure(1); plot(tempnew);
%tempnew