function adverse_testX = attack(testX, confidence)
load net;
adverse_testX = testX;                  % initialize
eta = 10;
B1 = .9;
B2 = .999;
eps = 10^(-8);
h = .5;
target = 5;
batch_size = 30;

for k = 1:3          % for all images in test set
  k
  conf_original = confidence(k, :)                % y = original confidence vals
  M = zeros(28^2,1); v = zeros(28^2,1); T = zeros(28^2,1);
  converged = false;
  iter = 1;
  all_loss = [];
  image_loss = [];
  conf_converge = [];
  while converged == false
      x = adverse_testX(:, :, 1, k);      % x = one image input
      x = double(get_vector(x));          % convert 2-d matrix to 1-d vector
      h_ = []; p = []; loss1 = []; loss2 = []; g = []; coord_x = []; coord_y = [];
      image_loss(iter) = loss_func(x, target, net);
      for L = 1:batch_size
          ei = zeros(28^2, 1);
          coord_x(L) = max([min([random('poiss', 14), 28]), 1]);      % x val of random pixel
          coord_y(L) = max([min([random('poiss', 14), 28]), 1]);      % y val of random pixel
          p(L) = 28 * (coord_x(L) - 1) + coord_y(L);   % random pixel value
          ei(p(L)) = 1;
          loss1(L) = loss_func(x + h*ei, target, net);
          loss2(L) = loss_func(x - h*ei, target, net);
          g(L) = (loss1(L) - loss2(L)) / (2*h);    % estimated gradient of that function
          h_(L) = (loss1(L) + loss2(L) - (2*image_loss(iter))) / (h^2);
      end
      % coord_x = max([min([random('poiss', 14), 28]), 1]);      % x val of random pixel
      % coord_y = max([min([random('poiss', 14), 28]), 1]);      % y val of random pixel
      % p = 28 * (coord_x - 1) + coord_y;   % random pixel value
      %%%%%%% [i, y2] = get_y2(y, target);      % y2 is confidence vals of not target
      % ei = zeros(28^2, 1);
      % ei(p) = 1;               % standard basis vector with only ith component as 1
      %image_loss(iter) = loss_func(x, target, net);
      %loss1 = loss_func(x + h*ei, target, net);
      %loss2 = loss_func(x - h*ei, target, net);
      %gi = (loss1 - loss2) / (2*h);    % estimated gradient of that function
      % if gi ~= 0
      % gi
      % end
      % hi =       % estimated second gradient
      for L = 1:batch_size
          if h_(L) <= 0
             delta_star = -eta*g(L);
          else
             delta_star = -eta*(g(L)/h_(L));
          end
          %T(p(L)) = T(p(L)) + 1;
          %M(p(L)) = B1*M(p(L)) + (1-B1)*g(L);
          %v(p(L)) = B2*v(p(L)) + (1-B2)*(g(L)^2);
          %M_hat_i = M(p(L)) / (1 - B1^(T(p(L))) );
          %v_hat_i = v(p(L)) / (1 - B2^(T(p(L))) );
          %delta_star = -eta * (M_hat_i / (sqrt(v_hat_i) + eps));
          adverse_testX(coord_y(L), coord_x(L), 1, k) = adverse_testX(coord_y(L), coord_x(L), 1, k) + delta_star;
      end
      %adverse_testX(coord_y, coord_x, 1, k) = adverse_testX(coord_y, coord_x, 1, k) + delta_star;
      conf_converge(iter, :) = predict(net, adverse_testX(:, :, 1, k));
      iter = iter + 1;
      if image_loss(iter-1) < .2 | iter == 700
          converged = true;
      end
  end
  if k < 10
      figure(k); plot(image_loss); title(['Targeted Loss for image', num2str(k)]);
      xlabel('Iteration of Attack Algorithm'); ylabel('Targeted Loss');
      figure(10+k); imshow(adverse_testX(:,:,:,k));
      figure(20+k); plot(conf_converge);
      title(['Confidence Values for image ',num2str(k)]);
      xlabel('Iteration of Attack Algorithm'); ylabel('Confidence Values');
      legend('Class 0', 'Class 1','Class 2','Class 3','Class 4','Class 5','Class 6', 'Class 7','Class 8','Class 9');
  end
  
  %plot(conf_converge); title(['Confidence Values for image ',num2str(k)]);
  %xlabel('Iteration of Attack Algorithm'); ylabel('Confidence Values');
  %legend('Class 0', 'Class 1','Class 2','Class 3','Class 4','Class 5','Class 6', 'Class 7','Class 8','Class 9');
  
  %conf_converge
  %conf_new = predict(net, adverse_testX(:, :, 1, k))
  
  %siz_img_loss = size(image_loss)
  %all_loss(k, 1:siz_img_loss(2)) = image_loss;
end

end

function loss = loss_func(x, target, net)
  % target = target + 1;
  K = .5;                         % transferability tuning parameter
  % y_target = double(y_target);
  img_temp = to_2D(x);  % convert back to 2d image
  %img_temp = repmat(img_temp,1,batch_size)
  new_prediction = predict(net, img_temp);
  conf_target = new_prediction(target);
  conf_not_target = [];
  for L = 0:9
      if L ~= target
          siz = size(conf_not_target);
          conf_not_target(siz(2)+1) = new_prediction(L+1);
      end
  end
  Fi = (exp(conf_not_target)) / (sum(conf_not_target) + conf_target);
  Ft = (exp(conf_target)) / (sum(conf_not_target) + conf_target);
  %Fi = (exp(y_not_target)) / (sum(y_not_target) + y_target);
  %Ft = (exp(y_target)) / (sum(y_not_target) + y_target);
  max1 = max(log10(Fi) - log10(Ft));
  loss = max([max1, -K]);
end

function img = to_2D(x)     % convert image vector to array
  img = zeros(28);
  for k = 1:28
      img(:,k) = x((28*(k-1)+1):(k*28));
  end
end

function x_vector = get_vector(image)
  x_vector = [];
  for k = 1:28
      low = 28*(k-1) + 1;
      high = 28*k;
      x_vector(low:high, 1) = image(:, k);
  end
end

function [i, y2] = get_y2(y, target)
  i = []; y2 = [];                    % i = classes that are not target
      for L = 0:9
         if L ~= target
             siz = size(i);
             i(siz(2)+1) = L;
             y2(siz(2)+1) = y(L+1);
         end
      end
end