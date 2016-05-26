%% Read Images from master folder
clear all;

tr_images_fld='/Users/rendanimbuvha/Dropbox/Image_Based_Recognition/Driver_Action_proj/imgs/train/';
classes=ls('/Users/rendanimbuvha/Dropbox/Image_Based_Recognition/Driver_Action_proj/imgs/train/');
count =1
for i=1:10    
      start=(1+(i-1)*3);      
      filePattern = fullfile(strcat(tr_images_fld,classes(start:start+1),'/'),'*.jpg');
      fileNames = dir(filePattern);
      for j=1:size(fileNames,1)
          if j>500
              break;
          end
          current_file=strcat(tr_images_fld,classes(start:start+1),'/',getfield(fileNames,{j},'name'));
          train_im{1,count}=single(imread(current_file));
          train_im{2,count}=i-1;
          count=count+1
     end
end
save('train_data.mat','train_im','-v7.3')
 