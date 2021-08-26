function berfin_kavsut_21602459_hw5(question)
clc
close all

switch question

case '1.1'

disp('1.1')
disp('Display the Data');

load('multicoil-random.mat');

%take 2DFT of image
for coil= 1:8
    d(:,:,coil) = fft2c(im(:,:,coil));
end

%SoS reconstruction 
m_sos = sos(im);
ref = m_sos;
%save reference image
save('reference.mat','ref');

%magnitude images for all coils 
figure;set(gcf, 'WindowState', 'maximized');
montage(abs(im),'DisplayRange', [], 'Size', [2 4]);
title('Magnitude Images for All Coils');
saveas(gcf,'1.1_mag.png');

%k-space spectrums for all coils 
figure; set(gcf, 'WindowState', 'maximized');
montage(log(abs(d)+1),'DisplayRange', [], 'Size', [2 4]);
title('K-Space Spectrums for All Coils');
saveas(gcf,'1.1_kspace.png');

%magnitude image for SoS
figure;set(gcf, 'WindowState', 'maximized');
imshow(abs(m_sos),[]);
title('Reference Image');
saveas(gcf,'1.1_sos.png');

case '1.2'
disp('1.2')
disp('Sampling Mask and Random Undersampled Images');

load('multicoil-random.mat');
load('reference.mat','ref');

%magnitude image for SoS
figure;set(gcf, 'WindowState', 'maximized');
imshow(log(abs(mask)+1),[]);
title('Undersampling Mask');
saveas(gcf,'1.2_mask.png');

%acceleration 
%R_overall = 1/(ratio of k-space acquired)
acquired_kspace = sum(mask(:));
all_kspace = numel(mask(:));
R_overall = 1/(acquired_kspace/all_kspace);
disp(strcat('The overall acceleration corresponding to the given mask:',num2str(R_overall)));

%take 2DFT of image, and undersample k-space datas with mask 
[datau,imu] = undersample(im,mask);

%k-space spectrums for all coils 
figure; set(gcf, 'WindowState', 'maximized');
montage(log(abs(datau)+1),'DisplayRange', [], 'Size', [2 4]);
title('Undersampled K-Space Spectrums for All Coils');
saveas(gcf,'1.2_kspace.png');

%magnitude images for all coils 
figure;set(gcf, 'WindowState', 'maximized');
montage(abs(imu),'DisplayRange', [], 'Size', [2 4]);
title('Undersampled Magnitude Images for All Coils');
saveas(gcf,'1.2_mag.png');

%SoS reconstruction 
m_sos = sos(imu);

%magnitude image for SoS
figure;set(gcf, 'WindowState', 'maximized');
imshow(abs(m_sos),[]);
title('Final Zero-Fill Image');
saveas(gcf,'1.2_sos.png');

%normalize reference image 
ref = abs(ref);
ref = ref/max(ref(:));

%normalize sos result image 
m_sos = abs(m_sos); % actually, m_sos is already real-valued image 
m_sos = m_sos/max(m_sos(:));

%error image
figure;set(gcf, 'WindowState', 'maximized');
imshow(abs(m_sos-ref), []);
title('Error Image');
saveas(gcf,'1.2_error.png');

%IQA results 
PSNR= psnr(m_sos,ref);
SSIM= ssim(m_sos,ref);
disp(strcat('PSNR:',num2str(PSNR)));
disp(strcat('SSIM:',num2str(SSIM)));

case '1.3'
disp('1.3')
disp('SPIRiT Kernel Weights');

load('multicoil-random.mat');

[data_calib] = imcalib(im, calib);
lambda = 0;
kernel =calibrateSpirit(data_calib,lambda);

%k-space spectrums of calibration regions from all coils 
figure;set(gcf, 'WindowState', 'maximized');
montage(log(abs(data_calib)+1),'DisplayRange', [], 'Size', [2 4], 'Border', 5);
title('Calibration Regions from All Coils');
saveas(gcf,'1.3_calib.png');

%magnitude imagepf SPIRiT kernel 
figure;
imshow(abs(kernel),[]);
title('Kernel');
saveas(gcf,'1.3_kernel.png');
size_kernel = size(kernel);
disp(strcat('size of kernel:',num2str(size_kernel(1)),'x',num2str(size_kernel(2))));


case '1.4'
disp('1.4')
disp('SPIRiT Reconstruction')

load('multicoil-random.mat');
load('reference.mat');

lambda = 1e-1; % by trial-and-error 

%spirit reconstruction 
[datau,imu] = undersample(im,mask);
[data_calib] = imcalib(im, calib);
kernel =calibrateSpirit(data_calib,lambda);
imr = spirit(datau, kernel);

%sos reconstruction of coil images found by spirit reconstruction  
m_sos = sos(imr);

%magnitude images for all coils by SPIRiT reconstruction 
figure;set(gcf, 'WindowState', 'maximized');
montage(abs(imr),'DisplayRange', [], 'Size', [2 4]);
title('SPIRiT Reconstruction for All Coils');
saveas(gcf,'1.4_spirit.png');

%normalize reference image 
ref = abs(ref);
ref = ref/max(ref(:));

%normalize sos result image 
m_sos = abs(m_sos); % actually, m_sos is already real-valued image 
m_sos = m_sos/max(m_sos(:));

%IQA results 
PSNR= psnr(m_sos,ref);
SSIM= ssim(m_sos,ref);
disp(strcat('PSNR:',num2str(PSNR)));
disp(strcat('SSIM:',num2str(SSIM)));

%reference image
figure;set(gcf, 'WindowState', 'maximized');
subplot(1,3,1); imshow(abs(ref), []);
title('Reference Image');
%SoS reconstruction 
subplot(1,3,2); imshow(abs(m_sos), []);
title('SPIRiT Reconstruction');
%error image 
subplot(1,3,3);imshow(abs(m_sos-ref), []);
title('Error Image');
saveas(gcf,'1.4.png');

case '1.5' % not finished yet!
disp('1.5');
disp('l1 Regularization in Wavelet Domain');

load('multicoil-random.mat');

load('reference.mat');
%normalize reference image 
ref = abs(ref);
ref = ref/max(ref(:));

%undersampled images for all coils 
[datau,imu] = undersample(im,mask);

%generate wavelet operator
wv = Wavelet('Daubechies',4,4); 
for i = 1:8
    imr = imu(:,:,i);
    imwv(:,:,i) = wv*imr; % take forward wavelet transform  
end 

%wavelet transform images for all coils  
figure;set(gcf, 'WindowState', 'maximized');
montage(log(abs(imwv)+1),'Size', [2 4], 'BorderSize', 2,'BackgroundColor', 'blue');
title('Wavelet Transforms for All Coil Images');
saveas(gcf,'1.5_wavelet.png');    
    
%different beta values to see its effect on reconstruction images
for beta = [0.01, 0.1, 1]
    
    %apply l1 regularization in wavelet domain
    for i = 1:8
        imr = imu(:,:,i);
        imth(:,:,i) = l1wavelet(imr,beta);
    end 
    
    %magnitude images from all coils after tresholding  
    figure;set(gcf, 'WindowState', 'maximized');
    montage(abs(imth),'Size', [2 4], 'BorderSize', 2,'BackgroundColor', 'blue');
    title(strcat('Coil Images after Tresholding in Wavelet Domain, Beta=',num2str(beta)));
    saveas(gcf,strcat('1.5_coils_beta',num2str(beta),'.png')); 

    %SoS reconstruction 
    m_sos = sos(imth);

    %normalize sos result image 
    m_sos = abs(m_sos); % actually, m_sos is already real-valued image 
    m_sos = m_sos/max(m_sos(:));

    %IQA results 
    PSNR = psnr(ref,m_sos);
    SSIM = ssim(ref,m_sos);
    disp(strcat('PSNR:',num2str(PSNR),', Beta:',num2str(beta)));
    disp(strcat('SSIM:',num2str(SSIM),', Beta:',num2str(beta)));

    %reference image
    figure;set(gcf, 'WindowState', 'maximized');
    subplot(1,3,1); imshow(ref, []);
    title('Reference Image');
    %SoS reconstruction 
    subplot(1,3,2); imshow(m_sos, []);
    title(strcat('Beta=',num2str(beta)));
    %error image 
    subplot(1,3,3);imshow(abs(m_sos-ref), []);
    title('Error Image');
    saveas(gcf,strcat('1.5_beta',num2str(beta),'.png'));

end 

case '1.6'
disp('1.6')
disp('Iterative l1-SPIRiT Reconstruction');

load('multicoil-random.mat');

[datau,imu] = undersample(im,mask); %undersampled images from each coil 
[data_calib] = imcalib(im, calib); %calibration region  from each coil 
lambda = 1e-1; %regularization parameter
kernel = calibrateSpirit(data_calib,lambda); %kernel calculated for each coil 

%Iterative l1-SPIRiT Reconstruction
beta = 0.001; %by trial-and-error 
numiter = 20; %iteration number 
imr = l1spirit(datau,kernel, beta, numiter);

disp(strcat('Beta: ',num2str(beta)) );

%Outputs are computed inside l1spirit function!

end

end

function imr = l1spirit(datau,kernel, beta, numiter)
% l1spirit function 
%
% inputs 
% datau : randomly undersampled k-space data 
% kernel : 
% beta : treshold in wavelet domain 
% numiter : iteration number
%
% outputs 
% imr : iterative l1-SPIRiT reconstruction image  
%
% STEPS 
% 1. Apply SPIRiT to reconstruct full k-space images for each coil.
% 2. Apply l1 regularization in wavelet domain to each image. 
% Choose a reasonable ? (by trial and error), and keep it constant across iterations.
% 3. Enforce data consistency in k-space by replacing the reconstructed 
% k-space data with the acquired data at sampled k-space locations.

load('reference.mat');
%normalize reference image 
ref = abs(ref);
ref = ref/max(ref(:));

[Nx Ny Nc] = size(datau);

kspace = datau; % first iteration, undersampled acquired k-space data 

for iter = 1:numiter
    
    %spirit reconstruction 
    imr = spirit(kspace, kernel);
    
    %l1 regularization in wavelet domain 
    for coil = 1:Nc
        %wavelet regularized image
        imth(:,:,coil) = l1wavelet(imr(:,:,coil),beta);
        %k-space data of the wavelet regularized image 
        kspace(:,:,coil) = fft2c(imth(:,:,coil));
    end   
   
    %data consistency
    %replace the acquired (i.e. non-zero points) 
    %from acquired randomly undersampled data (datau)    
    kspace(datau~=0) = datau(datau~=0);
    
    %take inverse 2DFT 
    for coil = 1:Nc
        imr(:,:,coil) = ifft2c(kspace(:,:,coil));
    end 
    
    %final SoS reconstrucion 
    m_sos = sos(imr);

    %normalize sos result image 
    m_sos = abs(m_sos); % actually, m_sos is already real-valued image 
    m_sos = m_sos/max(m_sos(:));
    
    %reference image
    figure;set(gcf, 'WindowState', 'maximized');
    subplot(1,3,1); imshow(ref, []);
    title('Reference Image');
    %SoS reconstruction 
    subplot(1,3,2); imshow(m_sos, []);
    title(strcat('Iterative l1-SPIRiT Reconstruction, Iteration#', num2str(iter)));
    %error image 
    subplot(1,3,3);imshow(abs(m_sos-ref), []);
    title('Error Image');
    saveas(gcf,strcat('1.6_iter#',num2str(iter),'.png'));

    PSNR(iter) = psnr(ref,m_sos);
    SSIM(iter) = ssim(ref,m_sos);
    disp(strcat('PSNR for Iteration#',num2str(iter),':',num2str(PSNR(iter))));
    disp(strcat('SSIM for Iteration#',num2str(iter),':',num2str(SSIM(iter))));

end

%PSNR and SSIM values vs. Iteration Number Plots 
figure;set(gcf, 'WindowState', 'maximized');
subplot(1,2,1); plot((1:numiter),PSNR); title("PSNR vs. Iteration #");
xlabel('Iteration #');ylabel('PSNR Value');
subplot(1,2,2); plot((1:numiter),SSIM); title("SSIM vs. Iteration #");
xlabel('Iteration #');ylabel('SSIM Value');
saveas(gcf,'1.6_psnr_ssim.png');

end 

function imth = l1wavelet(imr,beta)
% l1wavelet function 
%
% inputs 
% imr : reconstructed 2D image 
% beta : treshold in wavelet domain 
%
% outputs 
% imth :  wavelet regularized imr with beta constant
%

wv = Wavelet('Daubechies',4,4); % generate wavelet operator
coeffW = wv*imr; % take forward wavelet transform
%l1 regularization by soft tresholding
%apply soft tresholding in each elet coefficient 
S_beta = (coeffW./(abs(coeffW))).*max(0,abs(coeffW)-beta);
imth = wv'*S_beta; % take inverse wavelet transform

end 


function imr = spirit(datau, kernel)

% spirit function 
% this function applies spirit reconstruction without data consistency 
%
% inputs 
% datau : undersampled k-space data with calibration region 
% kernel : spirit kernel weights
%
% outputs 
% imr : size of (Nx x Ny x Nc) 
%
% pre-condition : (calibx,caliby) = (32,32)

%calibration region 
calibx = 32; caliby = 32;

%take sizes of k-space data 
[Nx Ny Nc] = size(datau); 

%set Mr to zero matrix  
Mr = zeros(Nx, Ny, Nc); 

%find spirit reconstruction for all coils 
for k = 1:Nc
    for col = 2:Ny-1 
        for row = 2:Nx-1
            
            %take coefficients for one coil from kernel, 
            %each column is for one coil 
            kernel_temp = kernel(:,k);
            
            %multiplications of samples in neighbourhood with 
            %their corresponding coefficients 
            sum = 0;          
            
            %go thorugh 71 samples in the neighbourhood of 
            %the calculated data point 
            ind = 1;
            
            %take coefficients calculated for all coils          
            for coil = 1:Nc      
                
                for j = [-1 0 1]
                    for i = [-1 0 1]
                        
                        %take 8 samples from the same coil
                        %take 9 samples from other coils 
                        if( ~( (coil==k) && (i==0 && j== 0) ) ) 
                            
                            %take corresponding coefficient 
                            %for the sample point
                            
                            coefficient = kernel_temp(ind);    
                            %sum up the multiplications of sample data point and 
                            %and their corresponding coefficients
                            
                            sum = sum + coefficient * datau(row+i,col+j,coil);
                            
                            %index for going through all coefficients 
                            ind = ind + 1;
                        end 
                    end 
                end                                          
            end    
            
            %put the found value to its position 
            Mr(row,col,k) = sum;   
            
        end 
    end
end 

for coil = 1:Nc
    %find spirit reconstructed images for all coils 
    imr(:,:,coil) = ifft2c(Mr(:,:,coil));   
end

end 

function kernel = calibrateSpirit(data_calib,lambda)

% calibrateSpirit function 
%
% inputs 
% data_calib : calibration region of  k-space data for all coils, size of(32x32)
% lambda : regularization parameter
%
% outputs 
% kernel :  2D matrix with size of ( 9 x Nc -1 ) x Nc
%
% pre-condition : (calibx,caliby) = (32,32)

[calibx caliby Nc] = size(data_calib);

Ma_coils = []; 

%collect coefficients for every coil, 
% and every data point inside calibration region of that coil 

for coil = 1:Nc
    
    Ma_coil = [];
    
    %collect coefficients for every data point inside calibration region,
    %excep for first and last row/column
    for col = 2:caliby-1
        for row = 2:calibx-1         
            
            Ma_row = zeros(1,9*Nc-1);  
            
            %go through 71 data points for one missing k-space data point
            ind = 0; 
            
            %take coefficients from all coils 
            for coil_no = 1:Nc
                for j = [-1 0  1]
                    for i = [-1 0 1]
                        %take 8 coefficients from the same coil
                        %take 9 coefficients from other coils 
                        if( ~( (coil_no==coil) && (i==0 && j== 0) ) ) 
                            ind = ind + 1; %increase index by 1
                            Ma_row(ind) = data_calib(row+i,col+j,coil_no);                            
                        end                    
                    end 
                end                 
            end   
            %repeat for every data point inside calibration region
            Ma_coil = [Ma_coil; Ma_row];
        end
    end
    %restore coefficient matrix for each coil separately 
    Ma_coils(:,:,coil) = Ma_coil;
end

%for each coil, we will find different kernels 
%each column of kernel will be to be used in one coil 
for k = 1:Nc

    %take data points for each coil from calibration region
    %create vector of data points 
    Mkc = [];    
    Mkc = data_calib((2:calibx-1),(2:caliby-1),k);
    Mkc = Mkc(:);
    
    %take coefficients for each coil 
    Ma = Ma_coils(:,:,k);
    
    %solve linear system of equations 
    %with regularized least squares estimation 
    ak = inv(Ma'*Ma + lambda*eye(size(Ma'*Ma))) * Ma' * Mkc;

    %set nan and inf values to zero 
    ak(isnan(ak)) = 0;
    ak(isinf(ak)) = 0;

    %each column of kernel has coefficients for each coil 
    a(:,k) = ak(:);
    
end

%set a to kernel
kernel = a;
 
end 

function [data_calib] = imcalib(im, calib)
%
% inputs 
% im : kspace data for all coils 
% calib : calibration region size, [calibx caliby]
%
% outputs 
% data_calib : calibration region data for all coils, 2D matrix with size of (32,32)
%

calibx = calib(1); caliby = calib(2); 

for coil_no = 1:8     
    %2D FFT 
    d = fft2c(im(:,:,coil_no));
    data_calib(:,:,coil_no) = d( (end/2 - calibx/2 + 1):(end/2 + calibx/2) , ...
                                 (end/2 - caliby/2 + 1):(end/2 + caliby/2) ); 
end 

end 

function [datau,imu] = undersample(im,mask)
    for coil= 1:8
        datau(:,:,coil) = fft2c(im(:,:,coil));
        datau(:,:,coil) = datau(:,:,coil).*mask;
        imu(:,:,coil) = ifft2c(datau(:,:,coil));
    end
end 

function  m = sos(im)

% sum of squares (sos) function 
% im: images from all coils 
% map: coil sensitivities
% m: result image

%map is ideally uniform over the entire object  
map = ones(size(im));
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


