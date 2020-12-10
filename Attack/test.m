load net;

%this is the network outputs when the inputs are the test images
outputs1 = predict(net, testX);

% **********************************
% ATTACK ***************************
testX_attacked = attack(testX, outputs1);
outputs2 = predict(net, testX_attacked);
var_pre_attack1 = var(outputs1(1,:))
var_post_attack1 = var(outputs2(1,:))
var_pre_attack2 = var(outputs1(2,:))
var_post_attack2 = var(outputs2(2,:))
var_pre_attack3 = var(outputs1(3,:))
var_post_attack3 = var(outputs2(3,:))
% **********************************
% **********************************

%this will give the predicted labels 
predLabelsTest = net.classify(testX_attacked);
%predLabelsTest = net.classify(testX);
%this gives the test accuracy
accuracy = sum(predLabelsTest == categorical(transpose(testY))) / numel(testY);

%plot a sample test image 
%figure(10000); imshow(testX_attacked(:,:,:,1));