function advers_testX = attack(testX)
adverse_testX = testX;

eta = .01;
B1 = .9;
B2 = .999;
eps = 10^(-8);

for i = 1:10000
  M = 0; v = 0; T = 0;
  converged = false;
  while converged == false
    coord_x = random('poiss', 14)      % x val of random pixel
    coord_y = random('poiss', 14)      % y val of random pixel
    F =        % DNN output for image x
    ei =       % standard basis vector with only ith component as 1
    fi1 =      % function of that DNN output
    fi2 =      % function of that DNN output
    gi =       % estimated gradient of that function
    hi =       % estimated second gradient
    T = T+1;
    M = B1*M + (1-B1)*gi;
    v = B2*v + (1-B2)*(gi^2);
    M_hat = M / (1-B1')
  end

end

end