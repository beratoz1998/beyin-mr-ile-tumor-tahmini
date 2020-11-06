

imds = imageDatastore('test', 'IncludeSubfolders',true, 'LabelSource','foldernames');
imds = shuffle(imds);
imds.ReadFcn=@readFCN;
 auimds = augmentedImageDatastore([224 224],imds);
load('trained_network.mat');

[YPredval,scoresval] = classify(netTransfer,auimds);

figure
for i=1:length(imds.Files)
    im=imread(imds.Files{i});
    subplot(4,5,i);
    imshow(im,'InitialMagnification','fit');
    title(cellstr(YPredval(i)));
end

%accuracy=sum(YPredval==imds.Labels)/length(imds.Files)
%yukarýdaki kodla sisteme öðrenmesi saðlandý. bütün datalar öðretildi
%ve çýktý resim olarak dosyaya eklendi.
