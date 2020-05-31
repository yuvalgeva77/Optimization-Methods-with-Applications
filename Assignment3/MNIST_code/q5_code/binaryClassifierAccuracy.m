function accuracy=binaryClassifierAccuracy(theta, X,y)

  y_hat = sigmoid(theta'*X) > 0.5;
  correct_labels = (y == y_hat);
  
%for eyes-on the mistaked examples
%   mistakes = X(:,correct_labels == 0);
%   mistake_labels = y(correct_labels == 0);
% % to view an example i as grayscale image- 
%   for i=1:length(mistake_labels)
%        figure;
%        imshow(vec2mat(mistakes(:,i),28));
%        title(['Label is: ',num2str(y(i))])
%   end
  correct=sum(correct_labels);
  accuracy = correct / length(y);