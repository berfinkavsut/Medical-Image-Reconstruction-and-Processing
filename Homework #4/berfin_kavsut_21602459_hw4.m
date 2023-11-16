function berfin_kavsut_21602459_hw4(question)
%clc
close all

switch question
    
%%part 1 - SENSE
case '1.1'
disp('1.1')

disp('Display the Data');
load('multicoil-data.mat');

%take 2DFT of image
for coil_no = 1:8
    d(:,:,coil_no) = fft2c(im(:,:,coil_no));
end 

%magnitude images for all coils 
figure;set(gcf, 'WindowState', 'maximized');
montage(abs(im),'DisplayRange', [], 'Size', [2 4], 'BorderSize', 5);
title('Magnitude Images for All Coils');
saveas(gcf,'1.1_mag.png');

%phase images for all coils 
figure;set(gcf, 'WindowState', 'maximized');
montage(angle(im),'DisplayRange', [], 'Size', [2 4], 'BorderSize', 5);
title('Phase Images for All Coils');
saveas(gcf,'1.1_phase.png');

%k-space spectrums for all coils 
figure; set(gcf, 'WindowState', 'maximized');
montage(log(abs(d)+1),'DisplayRange', [], 'Size', [2 4], 'BorderSize', 5);
title('K-Space Spectrums for All Coils');
saveas(gcf,'1.1_kspace.png');

%coil sensitivities for map1
%magnitudes of coil sensitivities 
figure;set(gcf, 'WindowState', 'maximized');
montage(abs(map1),'DisplayRange', [], 'Size', [2 4], 'BorderSize', 5);
title('Magnitudes of Coil Sensitivities (map1)');
saveas(gcf,'1.1_coil_mag_map1.png');

%phases of coil sensitivities 
figure;set(gcf, 'WindowState', 'maximized');
montage(angle(map1),'DisplayRange', [], 'Size', [2 4], 'BorderSize', 5);
title('Phases of Coil Sensitivities (map1)');
saveas(gcf,'1.1_coil_phase_map1.png');

%coil sensitivities for map2
%magnitudes of coil sensitivities 
figure;set(gcf, 'WindowState', 'maximized');
montage(abs(map2),'DisplayRange', [], 'Size', [2 4], 'BorderSize', 5);
title('Magnitudes of Coil Sensitivities (map2)');
saveas(gcf,'1.1_coil_mag_map2.png');

%phases of coil sensitivities 
figure;set(gcf, 'WindowState', 'maximized');
montage(angle(map2),'DisplayRange', [], 'Size', [2 4], 'BorderSize', 5);
title('Phases of Coil Sensitivities (map2)');
saveas(gcf,'1.1_coil_phase_map2.png');

case '1.2'
disp('1.2')
disp('Multi-coil Reconstruction');

load('multicoil-data.mat');

%SoS by map1
m_sos = sos(im,map1);

%OLC by map1
%reference image
m1_olc = olc(im,map1); 
%OLC by map2
m2_olc = olc(im,map2);

%magnitude image for SoS
figure;set(gcf, 'WindowState', 'maximized');
imshow(abs(m_sos),[]);
title('Magnitude Image for SoS by map1');
saveas(gcf,'1.2_sos.png');

%magnitude image for reference image (OLC by map1) 
figure;set(gcf, 'WindowState', 'maximized');
imshow(abs(m1_olc), []);
title('Magnitude Image for Reference Image (OLC by map1)');
saveas(gcf,'1.2_olc1.png');

%magnitude image for OLC by map2
figure;set(gcf, 'WindowState', 'maximized');
imshow(abs(m2_olc), []);
title('Magnitude Image for OLC by map2');
saveas(gcf,'1.2_olc2.png');

%normalize images
ref = m1_olc; 
ref = abs(ref); 
ref = ref/max(ref(:)); 
save('reference.mat','ref');

m_sos= abs(m_sos); 
norm_m_sos = m_sos/max(m_sos(:)); 

m2_olc= abs(m2_olc); 
norm_m2_olc = m2_olc/max(m2_olc(:));

%error image 
error_sos = abs(norm_m_sos-ref);
error_olc2 = abs(norm_m2_olc-ref);

%error image for SoS 
figure;set(gcf, 'WindowState', 'maximized');
imshow(abs(error_sos), []);
title('Error Image for SoS by map1');
saveas(gcf,'1.2_error_sos.png');

%magnitude image for OLC by map2
figure;set(gcf, 'WindowState', 'maximized');
imshow(abs(error_olc2), []);
title('Error Image for OLC by map2');
saveas(gcf,'1.2_error_olc2.png');

%IQA results 
PSNR_sos = psnr(norm_m_sos,ref);
SSIM_sos = ssim(norm_m_sos,ref);

PSNR_olc2 = psnr(norm_m2_olc,ref);
SSIM_olc2 = ssim(norm_m2_olc,ref);

disp(strcat('PSNR for SoS:',num2str(PSNR_sos)));
disp(strcat('SSIM for SoS:',num2str(SSIM_sos)));

disp(strcat('PSNR for OLC with map2:',num2str(PSNR_olc2)));
disp(strcat('SSIM for OLC with map2:',num2str(SSIM_olc2)));

case '1.3'
disp('1.3')

disp('Generate Undersampled Images');

load('multicoil-data.mat');

%(Rx,Ry) = (1,2)
Rx = 1; Ry = 2; 
[imu, Mu] = undersample(im, Rx, Ry);

%aliased images for all coils 
figure; set(gcf, 'WindowState', 'maximized');
montage(abs(imu),'displayRange',[], 'Size', [2 4], 'BorderSize', 5);
title('Magnitude Image for (Rx,Ry) = (1,2)');
saveas(gcf,'1.3_mag(1,2).png');

%undersampled k-space spectrums for all coils 
figure; set(gcf, 'WindowState', 'maximized');
montage(log(abs(Mu)+1),'displayRange',[], 'Size', [2 4], 'BorderSize', 5);
title('K-Spectrum for (Rx,Ry) = (1,2)');
saveas(gcf,'1.3_kspace(1,2).png');

%(Rx,Ry) = (2,1)
Rx = 2; Ry = 1; 
[imu, Mu] = undersample(im, Rx, Ry);

%aliased images for all coils 
figure; set(gcf, 'WindowState', 'maximized');
montage(abs(imu),'displayRange',[], 'Size', [2 4], 'BorderSize', 5);
title('Magnitude Image for (Rx,Ry) = (2,1)');
saveas(gcf,'1.3_mag(2,1).png');

%undersampled k-space spectrums for all coils 
figure; set(gcf, 'WindowState', 'maximized');
montage(log(abs(Mu)+1),'displayRange',[], 'Size', [2 4], 'BorderSize', 5);
title('K-Spectrum for (Rx,Ry) = (2,1)');
saveas(gcf,'1.3_kspace(2,1).png');

%(Rx,Ry) = (2,2)
Rx = 2; Ry = 2; 
[imu, Mu] = undersample(im, Rx, Ry);

%aliased images for all coils 
figure; set(gcf, 'WindowState', 'maximized');
montage(abs(imu),'displayRange',[], 'Size', [2 4], 'BorderSize', 5);
title('Magnitude Image for (Rx,Ry) = (2,2)');
saveas(gcf,'1.3_mag(2,2).png');

%undersampled k-space spectrums for all coils 
figure; set(gcf, 'WindowState', 'maximized');
montage(log(abs(Mu)+1),'displayRange',[], 'Size', [2 4], 'BorderSize', 5);
title('K-Spectrum for (Rx,Ry) = (2,2)');
saveas(gcf,'1.3_kspace(2,2).png');

%(Rx,Ry) = (1,4)
Rx = 1; Ry = 4; 
[imu, Mu] = undersample(im, Rx, Ry);

%aliased images for all coils 
figure; set(gcf, 'WindowState', 'maximized');
montage(abs(imu),'displayRange',[], 'Size', [1 8], 'BorderSize', 5);
title('Magnitude Image for (Rx,Ry) = (1,4)');
saveas(gcf,'1.3_mag(1,4).png');

%undersampled k-space spectrums for all coils 
figure; set(gcf, 'WindowState', 'maximized');
montage(log(abs(Mu)+1),'displayRange',[], 'Size', [1 8], 'BorderSize', 5);
title('K-Spectrum for (Rx,Ry) = (1,4)');
saveas(gcf,'1.3_kspace(1,4).png');

%(Rx,Ry) = (4,1)
Rx = 4; Ry = 1; 
[imu, Mu] = undersample(im, Rx, Ry);

%aliased images for all coils 
figure; set(gcf, 'WindowState', 'maximized');
montage(abs(imu),'displayRange',[], 'Size', [4 2], 'BorderSize', 5);
title('Magnitude Image for (Rx,Ry) = (4,1)');
saveas(gcf,'1.3_mag(4,1).png');

%undersampled k-space spectrums for all coils 
figure; set(gcf, 'WindowState', 'maximized');
montage(log(abs(Mu)+1),'displayRange',[], 'Size', [4 2], 'BorderSize', 5);
title('K-Spectrum for (Rx,Ry) = (4,1)');
saveas(gcf,'1.3_kspace(4,1).png');

case '1.4'
disp('1.4')
disp('SENSE Reconstruction');

load('multicoil-data.mat');
load('reference.mat');

%test inputs 
Rx = 1; Ry = 1;
lambda = 0;

[imu, Mu] = undersample(im, Rx, Ry);
im_sense = l2sense(imu, map1, Rx, Ry, lambda);

%normalize image
im_sense = abs(im_sense); 
im_sense = im_sense/max(im_sense(:)); 

%magnitude image for l2-regularized SENSE reconstruction 
figure; set(gcf, 'WindowState', 'maximized');
imshow(abs(im_sense),[]);
title('SENSE Reconstruction Magnitude Image (Rx,Ry)=(1,1), {\lambda}=0');
saveas(gcf,'1.4_mag.png');

%error image 
figure; set(gcf, 'WindowState', 'maximized');
imshow(abs(im_sense-ref),[]);
title('SENSE Reconstruction Error Image (Rx,Ry)=(1,1), {\lambda}=0');
saveas(gcf,'1.4_error.png');

%IQA results 
PSNR= psnr(im_sense,ref);
SSIM= ssim(im_sense,ref);

disp(strcat('PSNR for SENSE Reconstruction (Rx,Ry)=(1,1), lambda=0:',num2str(PSNR)));
disp(strcat('SSIM for SENSE Reconstruction (Rx,Ry)=(1,1), lambda=0:',num2str(SSIM)));

case '1.5'
disp('1.5');
disp('No Regularization');

load('multicoil-data.mat');
load('reference.mat');

%set regularization parameter to zero 
lambda = 0;

%(Rx,Ry) = (1,2)
Rx = 1; Ry = 2; 
[imu, Mu] = undersample(im, Rx, Ry);
im_sense = l2sense(imu, map1, Rx, Ry, lambda);

%normalize image
im_sense = abs(im_sense); 
im_sense = im_sense/max(im_sense(:)); 

%magnitude image
figure; set(gcf, 'WindowState', 'maximized');
imshow(abs(im_sense),[]);
title('Magnitude Image for (Rx,Ry) = (1,2)');
saveas(gcf,'1.5_mag(1,2).png');

%error image 
figure; set(gcf, 'WindowState', 'maximized');
imshow(abs(im_sense-ref),[]);
title('Error Image for (Rx,Ry) = (1,2)');
saveas(gcf,'1.5_error(1,2).png');

%IQA results 
PSNR0= psnr(im_sense,ref);
SSIM0= ssim(im_sense,ref);

%(Rx,Ry) = (2,1)
Rx = 2; Ry = 1; 
[imu, Mu] = undersample(im, Rx, Ry);
im_sense = l2sense(imu, map1, Rx, Ry, lambda);

%normalize image
im_sense = abs(im_sense); 
im_sense = im_sense/max(im_sense(:)); 

%magnitude image 
figure; set(gcf, 'WindowState', 'maximized');
imshow(abs(im_sense),[]);
title('Magnitude Image for (Rx,Ry) = (2,1)');
saveas(gcf,'1.5_mag(2,1).png');

%error image 
figure; set(gcf, 'WindowState', 'maximized');
imshow(abs(im_sense-ref),[]);
title('Error Image for (Rx,Ry) = (2,1)');
saveas(gcf,'1.5_error(2,1).png');

%IQA results 
PSNR1= psnr(im_sense,ref);
SSIM1= ssim(im_sense,ref);

%(Rx,Ry) = (2,2)
Rx = 2; Ry = 2; 
[imu, Mu] = undersample(im, Rx, Ry);
im_sense = l2sense(imu, map1, Rx, Ry, lambda);

%normalize image
im_sense = abs(im_sense); 
im_sense = im_sense/max(im_sense(:)); 

%magnitude image
figure; set(gcf, 'WindowState', 'maximized');
imshow(abs(im_sense),[]);
title('Magnitude Image for (Rx,Ry) = (2,2)');
saveas(gcf,'1.5_mag(2,2).png');

%error image 
figure; set(gcf, 'WindowState', 'maximized');
imshow(abs(im_sense-ref),[]);
title('Error Image for (Rx,Ry) = (2,2)');
saveas(gcf,'1.5_error(2,2).png');

%IQA results 
PSNR2= psnr(im_sense,ref);
SSIM2= ssim(im_sense,ref);

%(Rx,Ry) = (1,4)
Rx = 1; Ry = 4; 
[imu, Mu] = undersample(im, Rx, Ry);
im_sense = l2sense(imu, map1, Rx, Ry, lambda);

%normalize image
im_sense = abs(im_sense); 
im_sense = im_sense/max(im_sense(:)); 

%magnitude image
figure; set(gcf, 'WindowState', 'maximized');
imshow(abs(im_sense),[]);
title('Magnitude Image for (Rx,Ry) = (1,4)');
saveas(gcf,'1.5_mag(1,4).png');

%error image 
figure; set(gcf, 'WindowState', 'maximized');
imshow(abs(im_sense-ref),[]);
title('Error Image for (Rx,Ry) = (1,4)');
saveas(gcf,'1.5_error(1,4).png');

%IQA results 
PSNR3= psnr(im_sense,ref);
SSIM3= ssim(im_sense,ref);

%(Rx,Ry) = (4,1)
Rx = 4; Ry = 1; 
[imu, Mu] = undersample(im, Rx, Ry);
im_sense = l2sense(imu, map1, Rx, Ry, lambda);

%normalize image
im_sense = abs(im_sense); 
im_sense = im_sense/max(im_sense(:)); 

%magnitude image 
figure; set(gcf, 'WindowState', 'maximized');
imshow(abs(im_sense),[]);
title('Magnitude Image for (Rx,Ry) = (4,1)');
saveas(gcf,'1.5_mag(4,1).png');

%error image 
figure; set(gcf, 'WindowState', 'maximized');
imshow(abs(im_sense-ref),[]);
title('Error Image for (Rx,Ry) = (4,1)');
saveas(gcf,'1.5_error(4,1).png');

%IQA results 
PSNR4= psnr(im_sense,ref);
SSIM4= ssim(im_sense,ref);

disp(strcat('PSNR for SENSE Reconstruction (Rx,Ry)=(1,2):',num2str(PSNR0)));
disp(strcat('SSIM for SENSE Reconstruction (Rx,Ry)=(1,2):',num2str(SSIM0)));

disp(strcat('PSNR for SENSE Reconstruction (Rx,Ry)=(2,1):',num2str(PSNR1)));
disp(strcat('SSIM for SENSE Reconstruction (Rx,Ry)=(2,1):',num2str(SSIM1)));

disp(strcat('PSNR for SENSE Reconstruction (Rx,Ry)=(2,2):',num2str(PSNR2)));
disp(strcat('SSIM for SENSE Reconstruction (Rx,Ry)=(2,2):',num2str(SSIM2)));

disp(strcat('PSNR for SENSE Reconstruction (Rx,Ry)=(1,4):',num2str(PSNR3)));
disp(strcat('SSIM for SENSE Reconstruction (Rx,Ry)=(1,4):',num2str(SSIM3)));

disp(strcat('PSNR for SENSE Reconstruction (Rx,Ry)=(4,1):',num2str(PSNR4)));
disp(strcat('SSIM for SENSE Reconstruction (Rx,Ry)=(4,1):',num2str(SSIM4)));


case '1.6'
        
disp('1.6');
disp('SENSE Reconstruction With Regularization Parameter');

load('multicoil-data.mat');
load('reference.mat');

%set regularization parameter to 1 
lambda = 1;

%(Rx,Ry) = (1,2)
Rx = 1; Ry = 2; 
[imu, Mu] = undersample(im, Rx, Ry);
im_sense = l2sense(imu, map1, Rx, Ry, lambda);

%normalize image
im_sense = abs(im_sense); 
im_sense = im_sense/max(im_sense(:)); 

%magnitude image
figure; set(gcf, 'WindowState', 'maximized');
imshow(abs(im_sense),[]);
title('Magnitude Image for (Rx,Ry)=(1,2), {\lambda}=1');
saveas(gcf,'1.6_mag(1,2).png');

%error image 
figure; set(gcf, 'WindowState', 'maximized');
imshow(abs(im_sense-ref),[]);
title('Error Image for (Rx,Ry)=(1,2), {\lambda}=1');
saveas(gcf,'1.6_error(1,2).png');

%IQA results 
PSNR= psnr(im_sense,ref);
SSIM= ssim(im_sense,ref);
disp(strcat('PSNR for SENSE Reconstruction (Rx,Ry)=(1,2), lambda=1:',num2str(PSNR)));
disp(strcat('SSIM for SENSE Reconstruction (Rx,Ry)=(1,2), lambda=1:',num2str(SSIM)));

%(Rx,Ry) = (2,1)
Rx = 2; Ry = 1; 
[imu, Mu] = undersample(im, Rx, Ry);
im_sense = l2sense(imu, map1, Rx, Ry, lambda);

%normalize image
im_sense = abs(im_sense); 
im_sense = im_sense/max(im_sense(:)); 

%magnitude image 
figure; set(gcf, 'WindowState', 'maximized');
imshow(abs(im_sense),[]);
title('Magnitude Image for (Rx,Ry)=(2,1), {\lambda}=1');
saveas(gcf,'1.6_mag(2,1).png');

%error image 
figure; set(gcf, 'WindowState', 'maximized');
imshow(abs(im_sense-ref),[]);
title('Error Image for (Rx,Ry)=(2,1), {\lambda}=1');
saveas(gcf,'1.6_error(2,1).png');

%IQA results 
PSNR= psnr(im_sense,ref);
SSIM= ssim(im_sense,ref);
disp(strcat('PSNR for SENSE Reconstruction (Rx,Ry)=(2,1), lambda=1:',num2str(PSNR)));
disp(strcat('SSIM for SENSE Reconstruction (Rx,Ry)=(2,1), lambda=1:',num2str(SSIM)));

%(Rx,Ry) = (2,2)
Rx = 2; Ry = 2; 
[imu, Mu] = undersample(im, Rx, Ry);
im_sense = l2sense(imu, map1, Rx, Ry, lambda);

%normalize image
im_sense = abs(im_sense); 
im_sense = im_sense/max(im_sense(:)); 

%magnitude image
figure; set(gcf, 'WindowState', 'maximized');
imshow(abs(im_sense),[]);
title('Magnitude Image for (Rx,Ry)=(2,2), {\lambda}=1');
saveas(gcf,'1.6_mag(2,2).png');

%error image 
figure; set(gcf, 'WindowState', 'maximized');
imshow(abs(im_sense-ref),[]);
title('Error Image for (Rx,Ry)=(2,2), {\lambda}=1');
saveas(gcf,'1.6_error(2,2).png');

%IQA results 
PSNR= psnr(im_sense,ref);
SSIM= ssim(im_sense,ref);
disp(strcat('PSNR for SENSE Reconstruction (Rx,Ry)=(2,2), lambda=1:',num2str(PSNR)));
disp(strcat('SSIM for SENSE Reconstruction (Rx,Ry)=(2,2), lambda=1:',num2str(SSIM)));

%(Rx,Ry) = (1,4)
Rx = 1; Ry = 4; 
[imu, Mu] = undersample(im, Rx, Ry);
im_sense = l2sense(imu, map1, Rx, Ry, lambda);

%normalize image
im_sense = abs(im_sense); 
im_sense = im_sense/max(im_sense(:)); 

%magnitude image
figure; set(gcf, 'WindowState', 'maximized');
imshow(abs(im_sense),[]);
title('Magnitude Image for (Rx,Ry)=(1,4), {\lambda}=1');
saveas(gcf,'1.6_mag(1,4).png');

%error image 
figure; set(gcf, 'WindowState', 'maximized');
imshow(abs(im_sense-ref),[]);
title('Error Image for (Rx,Ry)=(1,4), {\lambda}=1');
saveas(gcf,'1.6_error(1,4).png');

%IQA results 
PSNR= psnr(im_sense,ref);
SSIM= ssim(im_sense,ref);
disp(strcat('PSNR for SENSE Reconstruction (Rx,Ry)=(1,4), lambda=1:',num2str(PSNR)));
disp(strcat('SSIM for SENSE Reconstruction (Rx,Ry)=(1,4), lambda=1:',num2str(SSIM)));

%(Rx,Ry) = (4,1)
Rx = 4; Ry = 1; 
[imu, Mu] = undersample(im, Rx, Ry);
im_sense = l2sense(imu, map1, Rx, Ry, lambda);

%normalize image
im_sense = abs(im_sense); 
im_sense = im_sense/max(im_sense(:)); 

%magnitude image 
figure; set(gcf, 'WindowState', 'maximized');
imshow(abs(im_sense),[]);
title('Magnitude Image for (Rx,Ry)=(4,1), {\lambda}=1');
saveas(gcf,'1.6_mag(4,1).png');

%error image 
figure; set(gcf, 'WindowState', 'maximized');
imshow(abs(im_sense-ref),[]);
title('Error Image for (Rx,Ry)=(4,1), {\lambda}=1');
saveas(gcf,'1.6_error(4,1).png');

%IQA results 
PSNR= psnr(im_sense,ref);
SSIM= ssim(im_sense,ref);
disp(strcat('PSNR for SENSE Reconstruction (Rx,Ry)=(4,1), lambda=1:',num2str(PSNR)));
disp(strcat('SSIM for SENSE Reconstruction (Rx,Ry)=(4,1), lambda=1:',num2str(SSIM)));

case '1.7'
        
disp('1.7');
disp('Importance of Accurate Coil Sensitivities');

load('multicoil-data.mat');
load('reference.mat');

%set regularization parameter to 1
lambda = 1;

%(Rx,Ry) = (1,2)
Rx = 1; Ry = 2; 
[imu, Mu] = undersample(im, Rx, Ry);
im_sense = l2sense(imu, map2, Rx, Ry, lambda);

%normalize image
im_sense = abs(im_sense); 
im_sense = im_sense/max(im_sense(:)); 

%magnitude image
figure; set(gcf, 'WindowState', 'maximized');
imshow(abs(im_sense),[]);
title('Magnitude Image for (Rx,Ry)=(1,2), {\lambda}=1');
saveas(gcf,'1.7_mag(1,2).png');

%error image 
figure; set(gcf, 'WindowState', 'maximized');
imshow(abs(im_sense-ref),[]);
title('Error Image for (Rx,Ry)=(1,2), {\lambda}=1');
saveas(gcf,'1.7_error(1,2).png');

%IQA results 
PSNR= psnr(im_sense,ref);
SSIM= ssim(im_sense,ref);
disp(strcat('PSNR for SENSE Reconstruction (Rx,Ry)=(1,2), lambda=1:',num2str(PSNR)));
disp(strcat('SSIM for SENSE Reconstruction (Rx,Ry)=(1,2), lambda=1:',num2str(SSIM)));

%(Rx,Ry) = (2,1)
Rx = 2; Ry = 1; 
[imu, Mu] = undersample(im, Rx, Ry);
im_sense = l2sense(imu, map2, Rx, Ry, lambda);

%normalize image
im_sense = abs(im_sense); 
im_sense = im_sense/max(im_sense(:)); 

%magnitude image 
figure; set(gcf, 'WindowState', 'maximized');
imshow(abs(im_sense),[]);
title('Magnitude Image for (Rx,Ry)=(2,1), {\lambda}=1');
saveas(gcf,'1.7_mag(2,1).png');

%error image 
figure; set(gcf, 'WindowState', 'maximized');
imshow(abs(im_sense-ref),[]);
title('Error Image for (Rx,Ry)=(2,1), {\lambda}=1');
saveas(gcf,'1.7_error(2,1).png');

%IQA results 
PSNR= psnr(im_sense,ref);
SSIM= ssim(im_sense,ref);
disp(strcat('PSNR for SENSE Reconstruction (Rx,Ry)=(2,1), lambda=1:',num2str(PSNR)));
disp(strcat('SSIM for SENSE Reconstruction (Rx,Ry)=(2,1), lambda=1:',num2str(SSIM)));

%(Rx,Ry) = (2,2)
Rx = 2; Ry = 2; 
[imu, Mu] = undersample(im, Rx, Ry);
im_sense = l2sense(imu, map2, Rx, Ry, lambda);

%normalize image
im_sense = abs(im_sense); 
im_sense = im_sense/max(im_sense(:)); 

%magnitude image
figure; set(gcf, 'WindowState', 'maximized');
imshow(abs(im_sense),[]);
title('Magnitude Image for (Rx,Ry)=(2,2), {\lambda}=1');
saveas(gcf,'1.7_mag(2,2).png');

%error image 
figure; set(gcf, 'WindowState', 'maximized');
imshow(abs(im_sense-ref),[]);
title('Error Image for (Rx,Ry)=(2,2), {\lambda}=1');
saveas(gcf,'1.7_error(2,2).png');

%IQA results 
PSNR= psnr(im_sense,ref);
SSIM= ssim(im_sense,ref);
disp(strcat('PSNR for SENSE Reconstruction (Rx,Ry)=(2,2), lambda=1:',num2str(PSNR)));
disp(strcat('SSIM for SENSE Reconstruction (Rx,Ry)=(2,2), lambda=1:',num2str(SSIM)));

%(Rx,Ry) = (1,4)
Rx = 1; Ry = 4; 
[imu, Mu] = undersample(im, Rx, Ry);
im_sense = l2sense(imu, map2, Rx, Ry, lambda);

%normalize image
im_sense = abs(im_sense); 
im_sense = im_sense/max(im_sense(:)); 

%magnitude image
figure; set(gcf, 'WindowState', 'maximized');
imshow(abs(im_sense),[]);
title('Magnitude Image for (Rx,Ry)=(1,4), {\lambda}=1');
saveas(gcf,'1.7_mag(1,4).png');

%error image 
figure; set(gcf, 'WindowState', 'maximized');
imshow(abs(im_sense-ref),[]);
title('Error Image for (Rx,Ry)=(1,4), {\lambda}=1');
saveas(gcf,'1.7_error(1,4).png');

%IQA results 
PSNR= psnr(im_sense,ref);
SSIM= ssim(im_sense,ref);
disp(strcat('PSNR for SENSE Reconstruction (Rx,Ry)=(1,4), lambda=1:',num2str(PSNR)));
disp(strcat('SSIM for SENSE Reconstruction (Rx,Ry)=(1,4), lambda=1:',num2str(SSIM)));

%(Rx,Ry) = (4,1)
Rx = 4; Ry = 1; 
[imu, Mu] = undersample(im, Rx, Ry);
im_sense = l2sense(imu, map2, Rx, Ry, lambda);

%normalize image
im_sense = abs(im_sense); 
im_sense = im_sense/max(im_sense(:)); 

%magnitude image 
figure; set(gcf, 'WindowState', 'maximized');
imshow(abs(im_sense),[]);
title('Magnitude Image for (Rx,Ry)=(4,1), {\lambda}=1');
saveas(gcf,'1.7_mag(4,1).png');

%error image 
figure; set(gcf, 'WindowState', 'maximized');
imshow(abs(im_sense-ref),[]);
title('Error Image for (Rx,Ry)=(4,1), {\lambda}=1');
saveas(gcf,'1.7_error(4,1).png');

%IQA results 
PSNR= psnr(im_sense,ref);
SSIM= ssim(im_sense,ref);
disp(strcat('PSNR for SENSE Reconstruction (Rx,Ry)=(4,1), lambda=1:',num2str(PSNR)));
disp(strcat('SSIM for SENSE Reconstruction (Rx,Ry)=(4,1), lambda=1:',num2str(SSIM)));

case '1.8'
disp('1.8')

disp('g-Factor for l2-regularized SENSE');
disp('Solution is on report!');
    
case '1.9'
disp('1.9');

disp('g-Factor Map');

load('multicoil-data.mat');

%set regularization parameter to 1
lambda = 1;

%map1 is used for g-factor 

%(Rx,Ry) = (1,2)
Rx = 1; Ry = 2; 
g = gfactor(map1, Rx, Ry, lambda);

%g-factor map
figure; set(gcf, 'WindowState', 'maximized');
imshow(real(g),'DisplayRange',[0 5]);
title('g-Factor Map for (Rx,Ry)=(1,2), {\lambda}=1');
saveas(gcf,'1.9_mag(1,2).png');

%(Rx,Ry) = (2,1)
Rx = 2; Ry = 1; 
g = gfactor(map1, Rx, Ry, lambda);

%g-factor map
figure; set(gcf, 'WindowState', 'maximized');
imshow(real(g),'DisplayRange',[0 5]);
title('g-Factor Map for (Rx,Ry)=(2,1), {\lambda}=1');
saveas(gcf,'1.9_mag(2,1).png');

%(Rx,Ry) = (2,2)
Rx = 2; Ry = 2; 
g = gfactor(map1, Rx, Ry, lambda);

%g-factor map
figure; set(gcf, 'WindowState', 'maximized');
imshow(real(g),'DisplayRange',[0 5]);
title('g-Factor Map for (Rx,Ry)=(2,2), {\lambda}=1');
saveas(gcf,'1.9_mag(2,2).png');

%(Rx,Ry) = (1,4)
Rx = 1; Ry = 4; 
g = gfactor(map1, Rx, Ry, lambda);

%g-factor map
figure; set(gcf, 'WindowState', 'maximized');
imshow(real(g),'DisplayRange',[0 5]);
title('g-Factor Map for (Rx,Ry)=(1,4), {\lambda}=1');
saveas(gcf,'1.9_mag(1,4).png');

%(Rx,Ry) = (4,1)
Rx = 4; Ry = 1; 
g = gfactor(map1, Rx, Ry, lambda);

%g-factor map
figure; set(gcf, 'WindowState', 'maximized');
imshow(real(g),'DisplayRange',[0 5]);
title('g-Factor Map for (Rx,Ry)=(4,1), {\lambda}=1');
saveas(gcf,'1.9_mag(4,1).png');

%%part2 - GRAPPA 

case '2.1'
disp('2.1');

disp('Calibration Images');

load('multicoil-data.mat');

%(calibx, caliby) = (8,8)
calibx = 8; caliby = 8; 
[imc, Mc] = imcalib(im, calibx, caliby);

figure; set(gcf, 'WindowState', 'maximized');
montage(abs(imc),'displayRange',[], 'Size', [2 4], 'BorderSize', 5);
title('Magnitude Images for Calibration Region (8x8)');
saveas(gcf,'2.1_mag(8,8).png');

figure; set(gcf, 'WindowState', 'maximized');
montage(log(abs(Mc)+1),'displayRange',[], 'Size', [2 4], 'BorderSize', 5);
title('K-Spectrums for Calibration Region (8x8)');
saveas(gcf,'2.1_kspace(8,8).png');

%(calibx, caliby) = (16,16)
calibx = 16; caliby = 16; 
[imc, Mc] = imcalib(im, calibx, caliby);

figure; set(gcf, 'WindowState', 'maximized');
montage(abs(imc),'displayRange',[], 'Size', [2 4], 'BorderSize', 5);
title('Magnitude Images for Calibration Region (16x16)');
saveas(gcf,'2.1_mag(16,16).png');

figure; set(gcf, 'WindowState', 'maximized');
montage(log(abs(Mc)+1),'displayRange',[], 'Size', [2 4], 'BorderSize', 5);
title('K-Spectrums for Calibration Region (8x8)');
saveas(gcf,'2.1_kspace(16,16).png');

%(calibx, caliby) = (32,32)
calibx = 32; caliby = 32; 
[imc, Mc] = imcalib(im, calibx, caliby);

figure; set(gcf, 'WindowState', 'maximized');
montage(abs(imc),'displayRange',[], 'Size', [2 4], 'BorderSize', 5);
title('Magnitude Images for Calibration Region (32x32)');
saveas(gcf,'2.1_mag(32,32).png');

figure; set(gcf, 'WindowState', 'maximized');
montage(log(abs(Mc)+1),'displayRange',[], 'Size', [2 4], 'BorderSize', 5);
title('K-Spectrums for Calibration Region (32x32)');
saveas(gcf,'2.1_kspace(32,32).png');

case '2.2'
disp('2.2');
disp('Undersampled Images with Calibration Region');

load('multicoil-data.mat');

%(calibx,caliby) = (32,32)
calibx = 32; caliby = 32; 

%(Rx, Ry) = (1,2)
Rx = 1; Ry = 2;
[imu, Mu] = undersamplecalib(im, Rx, Ry, calibx, caliby);

figure; set(gcf, 'WindowState', 'maximized');
montage(abs(imu),'displayRange',[], 'Size', [2 4], 'BorderSize', 5);
title('Magnitude Images for (Rx,Ry) = (1,2)');
saveas(gcf,'2.2_mag(1,2).png');

figure; set(gcf, 'WindowState', 'maximized');
montage(log(abs(Mu)+1),'displayRange',[], 'Size', [2 4], 'BorderSize', 5);
title('K-Spectrums for (Rx,Ry) = (1,2)');
saveas(gcf,'2.2_kspace(1,2).png');

%(Rx, Ry) = (2,1)
Rx = 2; Ry = 1;
[imu, Mu] = undersamplecalib(im, Rx, Ry, calibx, caliby);
figure; set(gcf, 'WindowState', 'maximized');
montage(abs(imu),'displayRange',[], 'Size', [2 4], 'BorderSize', 5);
title('Magnitude Images for (Rx,Ry) = (2,1)');
saveas(gcf,'2.2_mag(2,1).png');

figure; set(gcf, 'WindowState', 'maximized');
montage(log(abs(Mu)+1),'displayRange',[], 'Size', [2 4], 'BorderSize', 5);
title('K-Spectrums for (Rx,Ry) = (2,1)');
saveas(gcf,'2.2_kspace(2,1).png');

%(Rx, Ry) = (2,2)
Rx = 2; Ry = 2;
[imu, Mu] = undersamplecalib(im, Rx, Ry, calibx, caliby);
figure; set(gcf, 'WindowState', 'maximized');
montage(abs(imu),'displayRange',[], 'Size', [2 4], 'BorderSize', 5);
title('Magnitude Images for (Rx,Ry) = (2,2)');
saveas(gcf,'2.2_mag(2,2).png');

figure; set(gcf, 'WindowState', 'maximized');
montage(log(abs(Mu)+1),'displayRange',[], 'Size', [2 4], 'BorderSize', 5);
title('K-Spectrums for (Rx,Ry) = (2,2)');
saveas(gcf,'2.2_kspace(2,2).png');

%(Rx, Ry) = (1,4)
Rx = 1; Ry = 4;
[imu, Mu] = undersamplecalib(im, Rx, Ry, calibx, caliby);
figure; set(gcf, 'WindowState', 'maximized');
montage(abs(imu),'displayRange',[], 'Size', [2 4], 'BorderSize', 5);
title('Magnitude Images for (Rx,Ry) = (1,4)');
saveas(gcf,'2.2_mag(1,4).png');

figure; set(gcf, 'WindowState', 'maximized');
montage(log(abs(Mu)+1),'displayRange',[], 'Size', [2 4], 'BorderSize', 5);
title('K-Spectrums for (Rx,Ry) = (1,4)');
saveas(gcf,'2.2_kspace(1,4).png');

%(Rx, Ry) = (4,1)
Rx = 4; Ry = 1;
[imu, Mu] = undersamplecalib(im, Rx, Ry, calibx, caliby);

figure; set(gcf, 'WindowState', 'maximized');
montage(abs(imu),'displayRange',[], 'Size', [2 4], 'BorderSize', 5);
title('Magnitude Images for (Rx,Ry) = (4,1)');
saveas(gcf,'2.2_mag(4,1).png');

figure; set(gcf, 'WindowState', 'maximized');
montage(log(abs(Mu)+1),'displayRange',[], 'Size', [2 4], 'BorderSize', 5);
title('K-Spectrums for (Rx,Ry) = (4,1)');
saveas(gcf,'2.2_kspace(4,1).png');

case '2.3'
disp('2.3');
disp('GRAPPA Kernel Weights');

load('multicoil-data.mat');

%(Rx, Ry) = (1,2), (calibx,caliby) = (32,32), lambda=0

Rx = 1; Ry = 2;
lambda = 0;
calibx = 32; caliby = 32; 

[imc, Mc] = imcalib(im, calibx, caliby);
kernel = calibrate(Mc,Ry,lambda); % size : ( ( 6*Nc ) x Nc ) = ( 48 x 8 )
 
%display magnitude of kernel as an image
figure; set(gcf, 'WindowState', 'maximized');
imshow(abs(kernel),[]);
%title('Kernel for (Rx,Ry) = (1,2)');
saveas(gcf,'2.3_kernel(1,2).png');

case '2.4'
disp('2.4');
disp('GRAPPA Kernel Weights for (Rx,Ry) = (1,4)');

load('multicoil-data.mat');

%(Rx, Ry) = (1,4), (calibx,caliby) = (32,32), lambda=0
Rx = 1; Ry = 4;
lambda = 0;
calibx = 32; caliby = 32; 

[imc, Mc] = imcalib(im, calibx, caliby);
% kernel size : ( ( 10*Nc ) x Nc x (Ry-1) ) = ( 80 x 8 x 3 )
kernel = calibrate(Mc,Ry,lambda);
 
%display magnitude of kernel as an image
%for first line 
figure; set(gcf, 'WindowState', 'maximized');
subplot(1,3,1); imshow(abs(kernel(:,:,1)),[]);
title('First line');

%display magnitude of kernel as an image
%for second line
subplot(1,3,2); imshow(abs(kernel(:,:,2)),[]);
title('Second Line');

%display magnitude of kernel as an image
%for third line 
subplot(1,3,3); imshow(abs(kernel(:,:,3)),[]);
title('Third Line');
saveas(gcf,'2.4.png');

case '2.5'
disp('2.5');
disp('GRAPPA Reconstruction');

load('multicoil-data.mat');
 
%use SoS by map1 from Q1.2 as reference image
ref = sos(im,map1);
ref = abs(ref); 
ref = ref/max(ref(:));

%(Rx, Ry) = (1,2), (calibx,caliby) = (32,32), lambda=0
Ry = 2; Rx = 1;
calibx = 32; caliby = 32; 
lambda = 0;

[imu, Mu] = undersamplecalib(im, Rx, Ry, calibx, caliby);
[imc, Mc] = imcalib(im, calibx, caliby);
[imr, Mr] = grappa(Mu, Mc, Ry, lambda);
 
%magnitude images 
figure; set(gcf, 'WindowState', 'maximized');
montage(abs(imr),'displayRange',[], 'Size', [2 4], 'BorderSize', 5);
title('Magnitude Images of GRAPPA (Rx,Ry)=(1,2), {\lambda}=0');
saveas(gcf,'2.5_mag.png');

%k-space images 
figure; set(gcf, 'WindowState', 'maximized');
montage(log(abs(Mr)+1),'displayRange',[], 'Size', [2 4], 'BorderSize', 5);
title('K-Spectrums of GRAPPA (Rx,Ry)=(1,2), {\lambda}=0');
saveas(gcf,'2.5_kspace.png');

%final image of GRAPPA by SoS
map = ones(size(imr));
m = sos(imr,map);

%final image
figure; set(gcf, 'WindowState', 'maximized');
imshow(abs(m),[]);
title('Final Image of GRAPPA (Rx,Ry)=(1,2), {\lambda}=0');
saveas(gcf,'2.5_final.png');

%normalize final image
m = abs(m); 
m = m/max(m(:));

%error image 
figure; set(gcf, 'WindowState', 'maximized');
imshow(abs(m-ref),[]);
title('Error Image of GRAPPA (Rx,Ry)=(1,2), {\lambda}=0');
saveas(gcf,'2.5_error.png');

%IQA results 
PSNR= psnr(m,ref);
SSIM= ssim(m,ref);
disp(strcat('PSNR for GRAPPA Reconstruction (Rx,Ry)=(1,2), lambda=0:',num2str(PSNR)));
disp(strcat('SSIM for GRAPPA Reconstruction (Rx,Ry)=(1,2), lambda=0:',num2str(SSIM)));

%select lambda as regularization parameter
lambda = 1e12;

[imu, Mu] = undersamplecalib(im, Rx, Ry, calibx, caliby);
[imc, Mc] = imcalib(im, calibx, caliby);
[imr, Mr] = grappa(Mu, Mc, Ry, lambda);
 
%magnitude images 
figure; set(gcf, 'WindowState', 'maximized');
montage(abs(imr),'displayRange',[], 'Size', [2 4], 'BorderSize', 5);
title('Magnitude Images of GRAPPA (Rx,Ry)=(1,2), {\lambda}=1e12');
saveas(gcf,'2.5_mag_lambda.png');

%k-space images 
figure; set(gcf, 'WindowState', 'maximized');
montage(log(abs(Mr)+1),'displayRange',[], 'Size', [2 4], 'BorderSize', 5);
title('K-Spectrums of GRAPPA (Rx,Ry)=(1,2), {\lambda}=1e12');
saveas(gcf,'2.5_kspace_lambda.png');

%final image of GRAPPA by SoS
map = ones(size(imr));
m = sos(imr,map);

%final image
figure; set(gcf, 'WindowState', 'maximized');
imshow(abs(m),[]);
title('Final Image of GRAPPA (Rx,Ry)=(1,2), {\lambda}=1e12');
saveas(gcf,'2.5_final_lambda.png');

%normalize final image
m = abs(m); 
m = m/max(m(:));

%error image 
figure; set(gcf, 'WindowState', 'maximized');
imshow(abs(m-ref),[]);
title('Error Image of GRAPPA (Rx,Ry)=(1,2)');
saveas(gcf,'2.5_error_lambda.png');

%IQA results 
PSNR= psnr(m,ref);
SSIM= ssim(m,ref);
disp(strcat('PSNR for GRAPPA Reconstruction (Rx,Ry)=(1,2), lambda=1e12:',num2str(PSNR)));
disp(strcat('SSIM for GRAPPA Reconstruction (Rx,Ry)=(1,2), lambda=1e12:',num2str(SSIM)));


case '2.6'
    
disp('2.6');
disp('GRAPPA Reconstruction');

load('multicoil-data.mat');
 
%use SoS by map1 from Q1.2 as reference image
ref = sos(im,map1);
ref = abs(ref); 
ref = ref/max(ref(:));

%(Rx, Ry) = (1,4), (calibx,caliby) = (32,32), lambda=0
Ry = 4; Rx = 1;
calibx = 32; caliby = 32; 
lambda = 0;

[imu, Mu] = undersamplecalib(im, Rx, Ry, calibx, caliby);
[imc, Mc] = imcalib(im, calibx, caliby);
[imr, Mr] = grappa(Mu, Mc, Ry, lambda);
 
%magnitude images 
figure; set(gcf, 'WindowState', 'maximized');
montage(abs(imr),'displayRange',[], 'Size', [2 4], 'BorderSize', 5);
title('Magnitude Images of GRAPPA (Rx,Ry)=(1,4), {\lambda}=0');
saveas(gcf,'2.6_mag.png');

%k-space images 
figure; set(gcf, 'WindowState', 'maximized');
montage(log(abs(Mr)+1),'displayRange',[], 'Size', [2 4], 'BorderSize', 5);
title('K-Spectrums of GRAPPA (Rx,Ry)=(1,4), {\lambda}=0');
saveas(gcf,'2.6_kspace.png');

%final image of GRAPPA by SoS
map = ones(size(imr));
m = sos(imr,map);

%final image
figure; set(gcf, 'WindowState', 'maximized');
imshow(abs(m),[]);
title('Final Image of GRAPPA (Rx,Ry)=(1,4), {\lambda}=0');
saveas(gcf,'2.6_final.png');

%normalize final image
m = abs(m); 
m = m/max(m(:));

%error image 
figure; set(gcf, 'WindowState', 'maximized');
imshow(abs(m-ref),[]);
title('Error Image of GRAPPA (Rx,Ry)=(1,4), {\lambda}=0');
saveas(gcf,'2.6_error.png');

%IQA results 
PSNR= psnr(m,ref);
SSIM= ssim(m,ref);
disp(strcat('PSNR for GRAPPA Reconstruction (Rx,Ry)=(1,4), lambda=0:',num2str(PSNR)));
disp(strcat('SSIM for GRAPPA Reconstruction (Rx,Ry)=(1,4), lambda=0:',num2str(SSIM)));

%select lambda as regularization parameter
lambda = 1e14;

[imu, Mu] = undersamplecalib(im, Rx, Ry, calibx, caliby);
[imc, Mc] = imcalib(im, calibx, caliby);
[imr, Mr] = grappa(Mu, Mc, Ry, lambda);
 
%magnitude images 
figure; set(gcf, 'WindowState', 'maximized');
montage(abs(imr),'displayRange',[], 'Size', [2 4], 'BorderSize', 5);
title('Magnitude Images of GRAPPA (Rx,Ry)=(1,4), {\lambda}=1e14');
saveas(gcf,'2.6_mag_lambda.png');

%k-space images 
figure; set(gcf, 'WindowState', 'maximized');
montage(log(abs(Mr)+1),'displayRange',[], 'Size', [2 4], 'BorderSize', 5);
title('K-Spectrums of GRAPPA (Rx,Ry)=(1,4),{\lambda}=1e14');
saveas(gcf,'2.6_kspace_lambda.png');

%final image of GRAPPA by SoS
map = ones(size(imr));
m = sos(imr,map);

%final image
figure; set(gcf, 'WindowState', 'maximized');
imshow(abs(m),[]);
title('Final Image of GRAPPA (Rx,Ry)=(1,4), {\lambda}=1e14');
saveas(gcf,'2.6_final_lambda.png');

%normalize final image
m = abs(m); 
m = m/max(m(:));

%error image 
figure; set(gcf, 'WindowState', 'maximized');
imshow(abs(m-ref),[]);
title('Error Image of GRAPPA (Rx,Ry)=(1,4), {\lambda}=1e14');
saveas(gcf,'2.6_error_lambda.png');

%IQA results 
PSNR= psnr(m,ref);
SSIM= ssim(m,ref);
disp(strcat('PSNR for GRAPPA Reconstruction (Rx,Ry)=(1,4), lambda=1e14:',num2str(PSNR)));
disp(strcat('SSIM for GRAPPA Reconstruction (Rx,Ry)=(1,4), lambda=1e14:',num2str(SSIM)));

case '2.7'
disp('Final Remarks');
disp('Answer is on report!');

end

end


function [imr, Mr] = grappa(Mu, Mc, Ry, lambda)

% grappa function 
% inputs 
% Mu : undersampled k-space data with calibration region 
% Mc : k-space data with calibration region 
% Ry : acceleration rate on y-direction, geometry depends on Ry
% lambda - regularization parameter
%
% outputs 
% imr : size of (Nx x Ny x Nc) 
% Mr : size of (Nx x Ny x Nc) 
%
% pre-condition : Rx = 1
% pre-condition : (calibx,caliby) = (32,32)

Rx = 1;
calibx = 32; caliby = 32;

%take sizes of k-space data 
[Nx Ny Nc] = size(Mc); 

%find kernel to be used in grappa reconstruction 
kernel = calibrate(Mc,Ry,lambda);

%set Mr to undersampled k-psace data with calibration region 
Mr = Mu; 

if (Ry == 2)
%find grappa reconstruction for all coils 
for k = 1:Nc
    for col = [2:Ry:Ny-1] %skip one line in each iteration 
        for row = [2:Rx:Nx-1] %Rx = 1 

            %take kernel for one coil 
            kernel_temp = kernel(:,k);

            sum = 0;                
            index = 1;
            %take coefficients calculated for all coils 
            for coil = 1:Nc                   
                for j = [-1 1]
                    for i = [-1 0 1]
                        %take coefficient for the sample in neighbourhood
                        coefficient = kernel_temp(index);    
                        %sum up the multiplications of each
                        %coefficient and each sample
                        sum = sum + coefficient * Mu(row+i,col+j,coil);
                        %index for going through all coefficients 
                        index = index + 1;
                    end 
                end                                          
            end    
            %put found value to missing point
            Mr(row,col,k) = sum;                 
        end 
    end
end 
end 

if (Ry == 4)

for k = 1:Nc
    for col = 5:Ry:Ny-5
        for row =  3:1:Nx-2

        sum = 0;
        index = 1;
        kernel_temp = kernel(:,k,1); 
        for coil = 1:Nc
            ind = 1; 
            for j = [-1 3]
                for i = [-2 -1 0 1 2]
                    coefficient = kernel_temp(index);                           
                    sum = sum + coefficient * Mu(row+i,col+1+j,coil);
                    index = index + 1; 
                end 
            end 
        end
        Mr(row,col+1,k) = sum;

        sum = 0; 
        index = 1;
        kernel_temp = kernel(:,k,2); 
        for coil = 1:Nc
            for j = [-2 2]
                for i = [-2 -1 0 1 2]
                    coefficient = kernel_temp(index);                           
                    sum = sum + coefficient * Mu(row+i,col+2+j,coil);
                    index = index + 1; 
                end 
            end 
        end           
        Mr(row,col+2,k) = sum;

        sum=0;
        index = 1;
        kernel_temp = kernel(:,k,3);
        for coil = 1:Nc
           for j = [-3 1]
                for i = [-2 -1 0 1 2]
                    coefficient = kernel_temp(index);                           
                    sum = sum + coefficient * Mu(row+i,col+3+j,coil);
                    index = index + 1; 
                end 
            end 
        end
        Mr(row,col+3,k) = sum;  

        end 
    end
end    
end 

%correct calibration region 
for i = 1:Nc 
    
    %positions for calibration region 
    pos_calibx = (Nx/2 - calibx/2 +1): (Nx/2 + calibx/2);
    pos_caliby = (Ny/2 - caliby/2 +1): (Ny/2 + caliby/2);
    
    %set calibration region to original calibration measurement
    Mr(pos_calibx,pos_caliby,i) = Mu(pos_calibx,pos_caliby,i);
    
    %find grappa reconstructed images for all coils 
    imr(:,:,i) = ifft2c(Mr(:,:,i));    
end

end


function kernel = calibrate(Mc,Ry,lambda)

% calibrate function 
% inputs 
% Mc : k-space data with calibration region 
% Ry : acceleration rate on y-direction, geometry depends on Ry
% lambda - regularization parameter
%
% outputs 
% kernel : size of (6xNc), Nc = coil # 
%
% pre-condition : Rx = 1
% pre-condition : (calibx,caliby) = (32,32)

Rx = 1;
calibx = 32; caliby = 32;

%take sizes of k-space data 
[Nx Ny Nc] = size(Mc); 

%kernel for Ry = 2 
if (Ry == 2)
    
    %Ma will be constant for all coils 
    Ma = [];  
    %go through calibration region 
    %skip sample points for the points on first and last rows/columns 
    for col = (Ny/2 - caliby/2 + 1 + 1) : 1 : (Ny/2 + caliby/2 - 1)
        for row = (Nx/2 - calibx/2 + 1 + 1) : 1 : (Nx/2 + calibx/2 - 1) 
            
            %there are 6 samples in neighbourhood of our point
            %for each coil 
            temp = zeros(1,6);              
            %Ma_row is for 6 samples in neighbourhood of our point 
            %Ma_row has samples from all coils 
            Ma_row = [];
            for coil = 1:Nc
                ind = 1;  %for going through all sample points 
                for j = [-1 1]
                    for i = [-1 0 1]
                        %temp is for one point and one coil
                        temp(ind) = Mc(row+i,col+j,coil);
                        ind = ind + 1; 
                    end 
                end 
                %Ma_row is for one point and all coils 
                Ma_row = [Ma_row temp];            
            end  
            %Ma is for all points and all coils 
            Ma = [Ma; Ma_row];
        end 
    end
 
    %for each coil, we will find different kernels 
    %our result for kernel will contain coefficients to be used with all coils 
    for coil = 1:Nc
        
        Mkc = [];        
        %take all points in calibration region
        for col = (Ny/2 - caliby/2 + 1 + 1) : 1 : (Ny/2 + caliby/2 - 1)
            for row = (Nx/2 - calibx/2 + 1 + 1) : 1 : (Nx/2 + calibx/2 - 1) 
            temp = Mc(row,col,coil);
            %collect all data points inside Mkc 
            Mkc = [ Mkc; temp(:) ] ;              
            end 
        end 
        
        %for each coil, solve the linear system of equations 
        %for kernel coefficients 
        pseudo = Ma'*Ma;
        ak_temp = inv(pseudo + lambda*eye(size(pseudo))) * Ma' * Mkc;
        
        ak_temp(isnan(ak_temp)) = 0;
        ak_temp(isinf(ak_temp)) = 0;
        
        %each columns has coefficients for each coil 
        ak(:,coil) = ak_temp(:);
    end     
    kernel = ak; 
end 

%kernel for Ry = 4
if (Ry == 4)
    
    %Ma is constant for all coils
    %Ma is a 3d-matrix with Ma1, Ma2, and Ma3
    Ma1 = []; 
    Ma2 = []; 
    Ma3 = []; 
    
    %go through calibration region 
    %skip sample points for the points on first two and last two rows/columns 
    for col = (Ny/2 - caliby/2 + 1 + 2) : 1 : (Ny/2 + caliby/2 - 2)
        for row = (Nx/2 - calibx/2 + 1 + 2) : 1 : (Nx/2 + calibx/2 - 2) 
            
            %there are 10 samples in neighbourhood of our point
            %for each coil 
            temp = zeros(1,10);             
          
            %Ma_row1/Ma_row2/Ma_row3 is for 10 samples in neighbourhood of our point 
            %Ma_row1_Ma_row2_Ma_row3 has samples from all coils 
            Ma_row1 = [];
            Ma_row2 = [];
            Ma_row3 = [];
            
            for coil = 1:Nc
                ind = 1; %for going through all sample points 
                for j = [-1 3]
                    for i = [-2 -1 0 1 2]
                        %temp is for one point and one coil
                        temp(ind) = Mc(row+i,col+j,coil);
                        ind = ind + 1; 
                    end 
                end 
                %Ma_row1 is for one point and all coils 
                Ma_row1 = [Ma_row1 temp]; 
            end
            %Ma1 is for all points and all coils 
            Ma1 = [Ma1; Ma_row1];
             
            for coil = 1:Nc
                ind = 1;  %for going through all sample points 
                for j = [-2 2]
                    for i = [-2 -1 0 1 2]
                        %temp is for one point and one coil
                        temp(ind) = Mc(row+i,col+j,coil);
                        ind = ind + 1; 
                    end 
                end 
                %Ma_row2 is for one point and all coils 
                Ma_row2 = [Ma_row2 temp]; 
            end
            %Ma2 is for all points and all coils 
            Ma2 = [Ma2;Ma_row2];
             
            for coil = 1:Nc
                ind = 1;  %for going through all sample points 
                for j = [-3 1]
                    for i = [-2 -1 0 1 2]
                        %temp is for one point and one coil
                        temp(ind) = Mc(row+i,col+j,coil);
                        ind = ind + 1; 
                    end 
                end 
                %Ma_row2 is for one point and all coils 
                Ma_row3 = [Ma_row3 temp]; 
            end
            %Ma3 is for all points and all coils 
            Ma3 = [Ma3;Ma_row3];            
                                 
        end
    end
    
    %construct Ma
    %Ma is a 3d-matrix because each lines sees a different geometry
    Ma(:,:,1) =Ma1;
    Ma(:,:,2) =Ma2;
    Ma(:,:,3) =Ma3;
    
    %for each line, we have different geometry,
    %and we will solve them seperately 
    for l = 1:Ry-1
        
        %for each coil, we will find different kernels 
        %our result for kernel will contain coefficients 
        %to be used with all coils 
        for coil = 1:Nc
            
            Mkc = [];
            %take all points in calibration region
            for col = (Ny/2 - caliby/2 + 1 + 2) : 1 : (Ny/2 + caliby/2 - 2)
                for row = (Nx/2 - calibx/2 + 1 + 2) : 1 : (Nx/2 + calibx/2 - 2) %            
                    temp = Mc(row,col,coil);
                    %collect all data points inside Mkc 
                    Mkc = [ Mkc; temp(:) ] ;            
                end 
            end 
            
            %for each coil, solve the linear system of equations 
            %for kernel coefficients 
            pseudo = Ma(:,:,l)'*Ma(:,:,l);
            ak_temp= inv(pseudo + lambda*eye(size(pseudo))) * Ma(:,:,l)' * Mkc; 
           
            ak_temp(isnan(ak_temp)) = 0;
            ak_temp(isinf(ak_temp)) = 0;
            
            %each columns has coefficients for each coil 
            ak(:,coil,l) = ak_temp(:);
        end
        
    end 
    kernel = ak; 
end 
    

end


function [imu, Mu] = undersamplecalib(im, Rx, Ry, calibx, caliby)

% inputs 
% im : images from all coils 
% Rx : acceleration rate on x-direction
% Ry : acceleration rate on y-direction
% calibx : size of calibration region in x-direction 
% caliby : size of calibration ragion in y-direction 

% outputs 
% Mu : undersampled k-space data with calibration region
% imu : undersampled image with calibration region

%find calibration region 
[imc, Mc] = imcalib(im, calibx, caliby);

%set undersampled k-space data to undersampled k-space data 
Mu = Mc; 

%take sizes of k-space data
[Nkx Nky coil_no] = size(Mu);

%find positions to be set to Mu from full k-space data 
posx = 1:Rx:Nkx; % skip Rx-1 rows 
posy = 1:Ry:Nky; % skip Ry-1 columns 
    
for i = 1:coil_no 
    
    %take 2D FFT of image
    d = fft2c(im(:,:,i));
    
    %find undersampled k-space data with calibration region
    %set Mu from full k-space data at desired skipped positions 
    Mu(posx,posy,i) = d(posx,posy);
    %find undersampled image with calibration region
    imu(:,:,i) = ifft2c(Mu(:,:,i));
    
end

end


function [imc, Mc] = imcalib(im, calibx, caliby)

% inputs 
% im : images from all coils 
% calibx : size of calibration region in x-direction 
% caliby : size of calibration ragion in y-direction 

% outputs 
% Mc : k-space data of calibration region 
% imc : image computed with Mc

for coil_no = 1:8 
    
    %2D FFT 
    d = fft2c(im(:,:,coil_no));

    %k-space data will be zeroed out everywhere
    %except central region with size of (calibx x caliby)
    d(1:(end/2 - calibx/2),:) = 0;
    d((end/2 + calibx/2+1):end,:) = 0;
    d(:,1:(end/2 - caliby/2)) = 0;
    d(:,(end/2 + caliby/2+1):end) = 0;
 
    Mc(:,:,coil_no) = d; 
    imc(:,:,coil_no) = ifft2c(Mc(:,:,coil_no)); 
    
end 

end 

function g = gfactor(map, Rx, Ry, lambda)

% inputs 
% map : coil sensitivities from all coils 
% Rx : acceleration rate on x-direction
% Ry : acceleration rate on y-direction
% lambda : regularization parameter

% outputs 
% g : g-factor

%take sizes of coil sensitivities 
[Nx Ny coil_no] = size(map);

%take sizes for images to be aliased due to undersampling 
row_no = Nx/Rx; 
col_no = Ny/Ry; 

step_x = row_no; %increase source voxel indices as much as FOVx/Rx
step_y = col_no; %increase source voxel indices as much as FOVy/Ry

%go through all aliased points (xp)
for indx = 1:row_no
    for indy = 1:col_no
 
        %coil sensitivities for each xp
        C = []; 

        for x = 0:Rx-1
            for y = 0:Ry-1  
                
                %x-index of source voxel for xp
                indx_real = indx + Nx/2 - row_no/2 + (x * step_x);
                if (mod(indx_real, Nx)~=0)
                    indx_real = mod(indx_real, Nx); 
                end

                %y-index of source voxel for xp
                indy_real = indy + Ny/2 - col_no/2 + (y * step_y);
                if (mod(indy_real, Ny)~=0)
                    indy_real = mod(indy_real, Ny);
                end
                
                %take coil sensitivities at index of source voxel 
                temp = map(indx_real,indy_real,:);
                temp = temp(:); 
               
                %collect coil sensitivities for all source voxels in C 
                C = [C temp];
            end 
        end 
        
        index = 1;
        for x = 0:Rx-1
            for y = 0:Ry-1                
                
                %x-index of source voxel for xp
                indx_real = indx + Nx/2 - row_no/2 + (x * step_x);
                if (mod(indx_real, Nx)~=0)
                    indx_real = mod(indx_real, Nx); 
                end
                
                %y-index of source voxel for xp
                indy_real = indy + Ny/2 - col_no/2 + (y * step_y);
                if (mod(indy_real, Ny)~=0)
                    indy_real = mod(indy_real, Ny);
                end
                
                %take coil sensitivities corresponding to real image voxel 
                Cx = map(indx_real,indy_real,:);
                Cx = Cx(:);
                %find covariance from olc reconstruction
                cov_olc = Cx'*Cx; 
                
                %C is coil sensitivities for all source voxels for xp
                %find covariance from sense reconstruction
                cov_temp = inv(C'*C + lambda*eye(Rx*Ry));
                
                cov_temp(isnan(cov_temp)) = 0;
                cov_temp(isinf(cov_temp)) = 0;
                
                cov_sense = cov_temp * C' * C * cov_temp; 
                
                %if all of the original sensitivity maps have a value zero at a voxel,
                %set g-factor to 0 for that voxel
                if (sum(Cx(:))==0)
                    g_temp = 0;
                else
                    g_temp = sqrt( cov_olc * cov_sense(index,index) );       
                end 
                
                %put g-factor value to the index for the real image voxel 
                g(indx_real,indy_real) = g_temp; 
                %increment index in order to to go through all source voxels for xp 
                index = index + 1; 
            end 
        end 
    end 
end 

g(isnan(g)) = 0;
g(isinf(g)) = 0;

end 


function im_sense = l2sense(imu, map, Rx, Ry, lambda)

% inputs 
% imu : aliased images from all coils  
% map : coil sensitivities from all coils 
% Rx : acceleration rate on x-direction
% Ry : acceleration rate on y-direction
% lambda : regularization parameter

% outputs 
% im_sense : l2-regularized SENSE reconstruction image 

%take sizes of aliased image
[row_no col_no coil_no] = size(imu);
%take sizes for real image 
[Nx Ny coil_no] = size(map);

%result image
im_sense = zeros(Nx,Ny); %size : (FOVx)x(FOVy)

%when finding source voxels for the alised voxel,
%increase aliased voxel (xp) index by step size FOVx/Rx
step_x = row_no; 

%when finding source voxels for the alised voxel,
%increase aliased voxel (xp) index by step size FOVy/Ry
step_y = col_no; 

%compute image with all aliased points (xp)
%index of xp is (indx,indy)
for indx = 1:row_no
    for indy = 1:col_no

        %ms is from all coils at xp 
        ms = imu(indx,indy,:);
        ms = ms(:); 
        
        %C is coil sensitivities for each xp
        C = []; 

        %go through all source voxels for the aliased voxel xp 
        for x = 0:Rx-1
            for y = 0:Ry-1  
            
                %source voxel in x-direction 
                indx_real = indx + Nx/2 - row_no/2 + (x * step_x);
                if (mod(indx_real, Nx)~=0)
                    indx_real = mod(indx_real, Nx); 
                end

                %source voxel in y-direction
                indy_real = indy + Ny/2 - col_no/2 + (y * step_y);
                if (mod(indy_real, Ny)~=0)
                    indy_real = mod(indy_real, Ny);
                end
                
                %take coil sensitivities for one source voxel 
                temp = map(indx_real,indy_real,:);
                temp = temp(:); 
                
                % for one source voxel 
                % C : (coil #) x (source voxel #)
                C = [C temp];                
            end 
        end 

        %l2-regularized SENSE reconstruction 
        
        %m is source voxels for each xp, 
        %and we will solve our linear system of equations for m
        m = inv(C'*C + lambda*eye(Rx*Ry))*C'*ms;
        
        m(isnan(m)) = 0;
        m(isinf(m)) = 0;

        %put back calculated source voxels to their positions 
        %in result image im_sense in the order of positions where
        %we took source voxel sensitivities from coil sensitivities 
        for x = 0:Rx-1
            for y = 0:Ry-1
                
                %source voxel in x-direction 
                indx_real = indx + Nx/2 - row_no/2 + (x * step_x);
                if (mod(indx_real, Nx)~=0)
                    indx_real = mod(indx_real, Nx); 
                end

                %source voxel in y-direction
                indy_real = indy + Ny/2 - col_no/2 + (y * step_y);
                if (mod(indy_real, Ny)~=0)
                    indy_real = mod(indy_real, Ny);
                end
                im_sense(indx_real,indy_real) = m((x*Ry)+y+1);          
                
            end 
        end            

    end 
end 

im_sense(isnan(m)) = 0;
im_sense(isinf(m)) = 0;

end 

function [imu, Mu] = undersample(im, Rx, Ry)

% inputs 
% im : images from all coils  
% Rx : acceleration rate on x-direction
% Ry : acceleration rate on y-direction

% outputs 
% Mu : undersampled k-space data for all coils 
% imu : aliased images by undersampled k-space data for all coils 

for coil_no = 1:8 
    
    %take 2D FFT 
    d = fft2c(im(:,:,coil_no));
    [Nx Ny] = size(d);

    %generate positions to remain after undersampling 
    pos1 = 1:Rx:Nx; %rows (x-direction)
    pos2 = 1:Ry:Ny; %columns (y-direction)
    
    %discard k-space data with respect to Rx and Ry
    Mu(:,:,coil_no) = d(pos1,pos2); %size: (Nx/Rx) x (Ny/Ry) x Nc
    %take inverse 2D FFT
    imu(:,:,coil_no) = ifft2c(Mu(:,:,coil_no)); %size: (Nx/Rx) x (Ny/Ry) x Nc
    
end 

end

function m = olc(im,map)
% weighted linear combination (olc) function
% im: images from all coils 
% map: coil sensitivities 
% m: result image

%correct phase by complex conjugate of coil sensitivities  
phase_corrected = sum( conj(map).*im , 3 );
%correct weight by magnitude of cail sensitivities 
weight_corrected = ( 1./sum( abs(map).^2, 3 ) ).* phase_corrected; 
%result image
m= weight_corrected;

%correct image by setting inf and nan values to zero
m(isinf(m)) = 0;
m(isnan(m)) = 0; 

end 

function  m = sos(im,map)

% sum of squares (sos) function 
% im: images from all coils 
% map: coil sensitivities 
% m: result image

%find mss from coil images 
mss = sqrt( sum( im.*conj(im), 3) );
%find weights from coil sensitivities 
coil_sens= sqrt( sum( abs(map).^2,3 ) );
%find image from mss and calculated weights 
m = mss./coil_sens;

%correct image by setting inf and nan values to zero
m(isinf(m)) = 0;
m(isnan(m)) = 0;
    
end 

function d = fft2c(im)
% d = fft2c(im)
%
% fft2c performs a centered fft2
d = fftshift(fft2(ifftshift(im)));
end

function im = ifft2c(d)
% im = fft2c(d)
%
% ifft2c performs a centered ifft2
im = fftshift(ifft2(ifftshift(d)));
end
