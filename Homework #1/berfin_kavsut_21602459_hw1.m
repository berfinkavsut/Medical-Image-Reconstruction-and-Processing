function berfin_kavsut_21602459_hw1(question)
clc
close all

switch question
    case '1.1'        
        disp('1.1');
        disp('Full k-space Data');
        load('ge_phantom.mat');         
               
        [Nro Npe] = size(kdata);
        fprintf('Number of readout samples: %d\n',Nro);
        fprintf('Number of phase encode lines: %d\n',Npe);
        
        %display the k-space spectrum as an image
        figure;
        set(gcf, 'WindowState', 'maximized');
        imshow(log(abs(kdata)+1),[]);
        title('k-space spectrum');
        xlabel('ky'); ylabel('kx');           
        saveas(gcf,'1.1_k-space_data.png');
        
        %compute reference image
        %the reference image will be used for comparison 
        %in different reconstruction techniques
        mf = ifft2c(kdata); 
        ref = abs(mf);
        
        %display the magnitude image, phase image, 
        %real part of the image, and the imaginary part of the image
        figure;
        set(gcf, 'WindowState', 'maximized');
        subplot(2,2,1);        
        imshow(abs(mf),[]);
        title('Magnitude Image');
        xlabel('y'); ylabel('x');
        
        subplot(2,2,2);
        imshow(angle(mf),[]);
        title('Phase Image');
        xlabel('y'); ylabel('x');
        
        subplot(2,2,3);
        imshow(real(mf),[]);
        title('Real Part of Image');
        xlabel('y'); ylabel('x');
        
        subplot(2,2,4);
        imshow(imag(mf),[]);        
        title('Imaginary Part of Image');
        xlabel('y'); ylabel('x');
        saveas(gcf,'1.1_images.png');
        
    case '1.2'
        
        disp('1.2');
        disp('Phase-compensated Full k-space Image (±1/8th of k-space data)');
       
        load('ge_phantom.mat');          
        mf = ifft2c(kdata);  
        ref = abs(mf);
        
        %use ±1/8th of k-space data, which means
        %use the central 64 phase encode lines 
        ratio_symmetric = 1/8;
        [Ms,ms] = symmetric_kspace(kdata,ratio_symmetric);
        %phase-compensated full-kspace image
        m = real(phase_correct(mf,ms));
        %phase image
        p = exp(1i*angle(ms));
        %error image
        error_image = ref-m;
        
        figure;
        set(gcf, 'WindowState', 'maximized');
        %display the phase image, which is p(x,y)
        subplot(1,3,1);
        imshow(angle(p),[]);
        title('Phase Image');
        xlabel('y'); ylabel('x'); 
        
        %display the phase-compensated full k-space image, which is m
        subplot(1,3,2);
        imshow(m,[]);
        title('Phase-compensated Full k-space Image');
        xlabel('y'); ylabel('x');        
        
        %display the error image
        subplot(1,3,3);
        imshow(error_image,[]);
        title('Error Image');
        xlabel('y'); ylabel('x');
        saveas(gcf,'1.2_images.png');
                
        %normalize the reference image
        ref = ref./(max(max(ref)));
        %normalize the result image
        m = m./(max(max(m)));
        %IQA results 
        PSNR = psnr(m,ref);
        SSIM = ssim(m,ref);
        
        disp(strcat('PSNR:',num2str(PSNR)));
        disp(strcat('SSIM:',num2str(SSIM)));
        
    case '1.3'
        
        disp('1.3');
        disp('Phase-compensated Full k-space Image (±1/16th of k-space data)');
        load('ge_phantom.mat');          
        mf = ifft2c(kdata);  
        ref = abs(mf);
        
        
        %use ±1/16th of k-space data, which means
        %use the central 32 phase encode lines 
        ratio_symmetric = 1/16;
        [Ms,ms] = symmetric_kspace(kdata,ratio_symmetric);
        %phase-compensated full-kspace image
        m = real(phase_correct(mf,ms));
        %phase image
        p = exp(1i*angle(ms));
        %error image
        error_image = ref-m;
        
        figure;
        set(gcf, 'WindowState', 'maximized');
        %display the phase image, which is p(x,y)
        subplot(1,3,1);
        imshow(angle(p),[]);
        title('Phase Image');
        xlabel('y'); ylabel('x'); 
        
        %display the phase-compensated full k-space image, which is m
        subplot(1,3,2);
        imshow(m,[]);
        title('Phase-compensated Full k-space Image');
        xlabel('y'); ylabel('x');        
        
        %display the error image
        subplot(1,3,3);
        imshow(error_image,[]);
        title('Error Image');
        xlabel('y'); ylabel('x');
        saveas(gcf,'1.3_images.png');
        
        %normalize the reference image
        ref = ref./(max(max(ref)));
        %normalize the result image
        m = m./(max(max(m)));
        %IQA results 
        PSNR = psnr(m,ref);
        SSIM = ssim(m,ref);
       
        disp(strcat('PSNR:',num2str(PSNR)));
        disp(strcat('SSIM:',num2str(SSIM)));
        
    case '1.4'
        
        disp('1.4');
        disp('Partial k-space Data with 5/8th of k-space Data');
        load('ge_phantom.mat');        
        ratio_partial = 5/8;
        [Mpk,mpk] = partial_kspace(kdata,ratio_partial);
        
        figure;
        set(gcf, 'WindowState', 'maximized');
        %display the partial k-space spectrum 
        %this partial k-space data will be used in 1.5-1.8
        imshow(log(abs(Mpk)+1),[]);
        title('The 5/8th of k-space Data');
        xlabel('ky'); ylabel('kx');  
        saveas(gcf,'1.4_images.png');
        
    case '1.5'        
        disp('1.5');
        disp('Trivial Reconstruction');
        ratio_partial = 5/8;        
        ratio_symmetric = 1/8;
        [M,m,error_image,PSNR,SSIM] = trivial(ratio_partial,ratio_symmetric);
       
        figure;
        set(gcf, 'WindowState', 'maximized');
        %display the magnitude image
        subplot(1,2,1);
        imshow( abs(m),[]);
        title('Image of Trivial Reconstruction');
        xlabel('y'); ylabel('x');
        
        %display the error image
        subplot(1,2,2);
        imshow(error_image,[]);
        title('Error Image of Trivial Reconstruction');
        xlabel('y'); ylabel('x');
        saveas(gcf,'1.5_images.png');
        
        disp(strcat('PSNR:',num2str(PSNR)));
        disp(strcat('SSIM:',num2str(SSIM)));
        
    case '1.6'        
        disp('1.6');
        disp('PCCS (phase corrected and conjugate synthesis)');
        ratio_partial = 5/8;        
        ratio_symmetric = 1/8;
        [M,m,error_image,PSNR,SSIM] = pccs(ratio_partial,ratio_symmetric); 
        
        figure;
        set(gcf, 'WindowState', 'maximized');
        %display the resulting k-space spectrum
        subplot(1,3,1);
        imshow(log(abs(M)+1),[]);
        title('k-space Spectrum of PCCS');
        xlabel('ky'); ylabel('kx');    
        
        %display the magnitude image
        subplot(1,3,2);
        imshow(abs(m),[]);
        title('Image of PCCS');
        xlabel('y'); ylabel('x');    
        
        %display the error image
        subplot(1,3,3);
        imshow(error_image,[]);
        title('Error Image of PCCS');
        xlabel('y'); ylabel('x');
        saveas(gcf,'1.6_images.png');
        
        disp(strcat('PSNR:',num2str(PSNR)));
        disp(strcat('SSIM:',num2str(SSIM)));
        
    case '1.7'
        disp('1.7');
        disp('Homodyne Reconstruction (by using 0-1-2 step function)');
        ratio_partial = 5/8;   
        ratio_symmetric = 1/8;
        [M,m,error_image,weighted_Mpk,PSNR,SSIM] = homodyne(ratio_partial,ratio_symmetric);        
        
        figure;
        set(gcf, 'WindowState', 'maximized');
        %display the weighted Mpk(kx,ky)
        imshow(log(abs(weighted_Mpk)+1),[]);
        title('Weighted Mpk(kx,ky)');
        xlabel('ky'); ylabel('kx');    
        saveas(gcf,'1.7_weighted_Mpk.png');
        
        figure;
        set(gcf, 'WindowState', 'maximized');
        %display the resulting k-space spectrum
        subplot(1,3,1);
        imshow(log(abs(M)+1),[]);
        title('k-space Spectrum of Homodyne Reconstruction');
        xlabel('ky'); ylabel('kx');    
        
        %display the image
        subplot(1,3,2);
        imshow(abs(m),[]);
        title('Image of Homodyne Reconstruction');
        xlabel('y'); ylabel('x');    
        
        %display the error image
        subplot(1,3,3);
        imshow(error_image,[]);
        title('Error Image of Homodyne Reconstruction');
        xlabel('y'); ylabel('x');
        saveas(gcf,'1.7_images.png');
        
        disp(strcat('PSNR:',num2str(PSNR)));
        disp(strcat('SSIM:',num2str(SSIM)));
        
    case '1.8'
        disp('1.8');
        disp('POCS Reconstruction');
        ratio_partial = 5/8;       
        ratio_symmetric= 1/8;
        [M,m,error_image,PSNR,SSIM] = pocs(ratio_partial,ratio_symmetric,true);
        
        figure;
        set(gcf, 'WindowState', 'maximized');
        %display the resulting k-space spectrum
        subplot(1,3,1);
        imshow(log(abs(M)+1),[]);
        title('k-space Spectrum of POCS Reconstruction');
        xlabel('ky'); ylabel('kx');    
        
        %display the image
        subplot(1,3,2);
        imshow(abs(m),[]);
        title('Image of POCS Reconstruction');
        xlabel('y'); ylabel('x');    
        
        %display the error image
        subplot(1,3,3);
        imshow(error_image,[]);
        title('Error Image of POCS Reconstruction');
        xlabel('y'); ylabel('x');
        saveas(gcf,'1.8_result_images.png');   
        
        disp(strcat('PSNR:',num2str(PSNR)));
        disp(strcat('SSIM:',num2str(SSIM)));
        
    case '1.9'
        %compare the performance of trivial reconstruction, PCCS reconstruction,
        %homodyne reconstruction, and POCS reconstruction
        %with 9/16th of k-space data
        disp('1.9');
        disp('Comparison of 4 Reconstruction Techniques with 9/16th of k-space Data');
        ratio_partial = 9/16;       
        ratio_symmetric= 1/16;
        [M_trivial,m_trivial,error_trivial,PSNR_tri,SSIM_tri] = trivial(ratio_partial,ratio_symmetric);
        [M_pccs,m_pccs,error_pccs,PSNR_pccs,SSIM_pccs] = pccs(ratio_partial,ratio_symmetric);
        [M_homodyne,m_homodyne,error_homodyne,weighted_Mpk,PSNR_homo,SSIM_homo] = homodyne(ratio_partial,ratio_symmetric);
        [M_pocs,m_pocs,error_pocs,PSNR_pocs,SSIM_pocs]= pocs(ratio_partial,ratio_symmetric,false);

        %compare k-space spectrums 
        figure;
        set(gcf, 'WindowState', 'maximized');
        subplot(1,4,1);
        imshow(log(abs(M_trivial)+1),[]);
        title('k-space Spectrum of Trivial');
        xlabel('ky'); ylabel('kx');
        subplot(1,4,2);
        imshow(log(abs(M_pccs)+1),[]);
        title('k-space Spectrum of PCCS');
        xlabel('ky'); ylabel('kx');        
        subplot(1,4,3);
        imshow(log(abs(M_homodyne)+1),[]);
        title('k-space Spectrum of Homodyne');
        xlabel('ky'); ylabel('kx');        
        subplot(1,4,4);
        imshow(log(abs(M_pocs)+1),[]);
        title('k-space Spectrum of POCS');
        xlabel('ky'); ylabel('kx');
        saveas(gcf,'1.9_k_spectrum_images.png');
        
        %compare resulting images
        figure;
        set(gcf, 'WindowState', 'maximized');
        subplot(1,4,1);
        imshow(m_trivial,[]);
        title('Image of Trivial');
        xlabel('y'); ylabel('x');
        subplot(1,4,2);
        imshow(m_pccs,[]);
        title('Image of PCCS');
        xlabel('y'); ylabel('x');
        subplot(1,4,3);
        imshow(m_homodyne,[]);
        title('Image of Homodyne');
        xlabel('y'); ylabel('x');
        subplot(1,4,4);
        imshow(m_pocs,[]);
        title('Image of POCS');
        xlabel('y'); ylabel('x');
        saveas(gcf,'1.9_images.png');
    
        %compare error images
        figure;
        set(gcf, 'WindowState', 'maximized');
        subplot(1,4,1);
        imshow(error_trivial,[]);
        title('Error Image of Trivial');
        xlabel('ky'); ylabel('kx');
        subplot(1,4,2);
        imshow(error_pccs,[]);
        title('Error Image of PCCS');
        xlabel('ky'); ylabel('kx');        
        subplot(1,4,3);
        imshow(error_homodyne,[]);
        title('Error Image of Homodyne');
        xlabel('ky'); ylabel('kx');        
        subplot(1,4,4);
        imshow(error_pocs,[]);
        title('Error Image of POCS');
        xlabel('ky'); ylabel('kx');
        saveas(gcf,'1.9_error_images.png');
        
        disp('PSNR Results');
        disp('----------------------');
        disp(strcat('PSNR of Trivial Reconstruction:',num2str(PSNR_tri)));
        disp(strcat('PSNR of PCCS Reconstruction:',num2str(PSNR_pccs)));
        disp(strcat('PSNR of Homodyne Reconstruction:',num2str(PSNR_homo)));
        disp(strcat('PSNR of POCS Reconstruction:',num2str(PSNR_pocs)));
        disp('');
        
        disp('SSIM Results');
        disp('----------------------');
        disp(strcat('SSIM of Trivial Reconstruction:',num2str(SSIM_tri)));
        disp(strcat('SSIM of PCCS Reconstruction:',num2str(SSIM_pccs)));
        disp(strcat('SSIM of Homodyne Reconstruction:',num2str(SSIM_homo)));
        disp(strcat('SSIM of POCS Reconstruction:',num2str(SSIM_pocs)));
        
    case '1.10'
        %compare the performance of trivial reconstruction, PCCS reconstruction,
        %homodyne reconstruction, and POCS reconstruction
        %with 17/32th of k-space data        
        disp('1.10');
        disp('Comparison of 4 Reconstruction Techniques with 17/32th of k-space Data');
       
        ratio_partial = 17/32;       
        ratio_symmetric= 1/32;
        [M_trivial,m_trivial,error_trivial,PSNR_tri,SSIM_tri] = trivial(ratio_partial,ratio_symmetric);
        [M_pccs,m_pccs,error_pccs,PSNR_pccs,SSIM_pccs] = pccs(ratio_partial,ratio_symmetric);
        [M_homodyne,m_homodyne,error_homodyne,weighted_Mpk,PSNR_homo,SSIM_homo] = homodyne(ratio_partial,ratio_symmetric);
        [M_pocs,m_pocs,error_pocs,PSNR_pocs,SSIM_pocs]= pocs(ratio_partial,ratio_symmetric,false);

        %compare k-space spectrums 
        figure;
        set(gcf, 'WindowState', 'maximized');
        subplot(1,4,1);
        imshow(log(abs(M_trivial)+1),[]);
        title('k-space Spectrum of Trivial');
        xlabel('ky'); ylabel('kx');
        subplot(1,4,2);
        imshow(log(abs(M_pccs)+1),[]);
        title('k-space Spectrum of PCCS');
        xlabel('ky'); ylabel('kx');        
        subplot(1,4,3);
        imshow(log(abs(M_homodyne)+1),[]);
        title('k-space Spectrum of Homodyne');
        xlabel('ky'); ylabel('kx');        
        subplot(1,4,4);
        imshow(log(abs(M_pocs)+1),[]);
        title('k-space Spectrum of POCS');
        xlabel('ky'); ylabel('kx');
        saveas(gcf,'1.10_k_spectrum_images.png');
        
        %compare resulting images
        figure;
        set(gcf, 'WindowState', 'maximized');
        subplot(1,4,1);
        imshow(m_trivial,[]);
        title('Image of Trivial');
        xlabel('y'); ylabel('x');
        subplot(1,4,2);
        imshow(m_pccs,[]);
        title('Image of PCCS');
        xlabel('y'); ylabel('x');
        subplot(1,4,3);
        imshow(m_homodyne,[]);
        title('Image of Homodyne');
        xlabel('y'); ylabel('x');
        subplot(1,4,4);
        imshow(m_pocs,[]);
        title('Image of POCS');
        xlabel('y'); ylabel('x');
        saveas(gcf,'1.10_images.png');
    
        %compare error images
        figure;
        set(gcf, 'WindowState', 'maximized');
        subplot(1,4,1);
        imshow(error_trivial,[]);
        title('Error Image of Trivial');
        xlabel('ky'); ylabel('kx');
        subplot(1,4,2);
        imshow(error_pccs,[]);
        title('Error Image of PCCS');
        xlabel('ky'); ylabel('kx');        
        subplot(1,4,3);
        imshow(error_homodyne,[]);
        title('Error Image of Homodyne');
        xlabel('ky'); ylabel('kx');        
        subplot(1,4,4);
        imshow(error_pocs,[]);
        title('Error Image of POCS');
        xlabel('ky'); ylabel('kx');
        saveas(gcf,'1.10_error_images.png');
        
        disp('PSNR Results');
        disp('----------------------');
        disp(strcat('PSNR of Trivial Reconstruction:',num2str(PSNR_tri)));
        disp(strcat('PSNR of PCCS Reconstruction:',num2str(PSNR_pccs)));
        disp(strcat('PSNR of Homodyne Reconstruction:',num2str(PSNR_homo)));
        disp(strcat('PSNR of POCS Reconstruction:',num2str(PSNR_pocs)));
        disp('');
        
        disp('SSIM Results');
        disp('----------------------');
        disp(strcat('SSIM of Trivial Reconstruction:',num2str(SSIM_tri)));
        disp(strcat('SSIM of PCCS Reconstruction:',num2str(SSIM_pccs)));
        disp(strcat('SSIM of Homodyne Reconstruction:',num2str(SSIM_homo)));
        disp(strcat('SSIM of POCS Reconstruction:',num2str(SSIM_pocs)));
        
    case '1.11'
        disp('1.11');
        disp('End comments are on the pdf file.')
        
end

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

function [Ms,ms] = symmetric_kspace(kdata,ratio_symmetric)  
    [Nro Npe] = size(kdata);
    %create the window to have the symmetric part of k-space data
    filter_Ms = zeros(1,Npe);
    filter_Ms((Npe/2-Npe*ratio_symmetric):(Npe/2+Npe*ratio_symmetric-1))= 1;
    %multiplication of the k-space data with the window function 
    Ms = kdata.*filter_Ms;
    ms = ifft2c(Ms); 
end

function [Mpk,mpk] = partial_kspace(kdata,ratio_partial)  
    [Nro Npe] = size(kdata);
    %create the filter to have partial k-space data
    %with the given ration 
    filter_Mpk = zeros(1,Npe);
    filter_Mpk(1:Npe*ratio_partial-1)= 1;
    %multiplication of the k-space data with the filter
    Mpk = kdata.*filter_Mpk;
    mpk = ifft2c(Mpk);
end

function M = conjugate_symmetry(phase_corrected_kdata,ratio)
    [Nro Npe] = size(phase_corrected_kdata);
    %enforce the phase corrected k-space data
    %to have conjugate symmetry
    %for conjugate symmetry: X(kx,ky) = X*(-kx,-ky)
    temp = phase_corrected_kdata;
    %X(-kx,-ky)
    temp = fliplr(flipud(temp));
    %X*(-kx,-ky)
    temp = conj(temp);
    %fill the zeros of the partial phase corrected k-space data
    %with the elements of its conjugate symmetric 
    M = zeros(Nro,Npe);
    M(:,1:(Npe*ratio)) = phase_corrected_kdata(:,1:(Npe*ratio));
    M(:,(Npe*ratio+1):Npe) = temp(:,(Npe*ratio+1):Npe);
end 

function weighted_Mpk = weightFunc(Mpk,ratio_symmetric)
    [Nro Npe] = size(Mpk);
    %create the weighting function  
    %the weighting function is the 0-1-2 step function
    W = zeros(1,Npe);   
    W(1:(Npe/2-Npe*ratio_symmetric-1)) = 2;
    W((Npe/2-Npe*ratio_symmetric):(Npe/2+Npe*ratio_symmetric-1)) = 1;
    %multiplication of the partial k-space data with the weighting function
    weighted_Mpk = Mpk.*W;
end

function m = phase_correct (mpk,ms)
    %phase of ms(x,y)
    p = exp(j*angle(ms));
    %phase compensated full k-space image
    p_conjugate = exp(-j*angle(ms));
    m = mpk.*p_conjugate; 
end

function Mi = replace_kspace(Mi,M,ratio_partial)
    [Nro Npe] = size(Mi);
    Mi(:,ratio_partial*Npe:Npe)= M(:,ratio_partial*Npe:Npe);    
end 

function [M,m,error_image,PSNR,SSIM] = trivial(ratio_partial,ratio_symmetric)
    load('ge_phantom.mat');
    %reference image
    mf = ifft2c(kdata); 
    ref = abs(mf);    
    
    %partial k-space data with the desired ratio
    [Mpk,mpk] = partial_kspace(kdata,ratio_partial);
    %the symmetric part of the k-space data
    [Ms,ms] = symmetric_kspace(kdata,ratio_symmetric);
    
    %phase corrected partial image data 
    m = phase_correct(mpk,ms);
    %phase corrected partial k-space data
    M = fft2c(m);      
    %get rid of the added parts to the end of k-space data
    %by taking partial k-space data with the desired ratio
    [M,m] = partial_kspace(M,ratio_partial);
    
    %trivial reconstruction image
    %take real part of image as result image
    m = real(m);    
    %error image
    error_image = ref - m;  
    
    %normalize the reference image        
    ref = ref./(max(max(ref)));
    %normalize the result image
    m = m./(max(max(m)));
    %IQA results 
    PSNR = psnr(m,ref);
    SSIM = ssim(m,ref);
end

function [M,m,error_image,PSNR,SSIM] = pccs(ratio_partial,ratio_symmetric)
    load('ge_phantom.mat');
    %reference image
    mf = ifft2c(kdata); 
    ref = abs(mf);
    
    %partial k-space data with the desired ratio
    [Mpk,mpk] = partial_kspace(kdata,ratio_partial);
    %the symmetric part of the k-space data
    [Ms,ms] = symmetric_kspace(kdata,ratio_symmetric);
     
    %phase corrected partial image data 
    phase_corrected_image = phase_correct(mpk,ms);
    %phase corrected partial k-space data 
    phase_corrected_kdata= fft2c(phase_corrected_image);
    %enforce conjugate symmetry to the phase corrected partial k-space data
    M = conjugate_symmetry(phase_corrected_kdata,ratio_partial);
    
    %PCCS result image
    %take real part of the image as result image
    m = real(ifft2c(M));
    %error image
    error_image = ref-m;
    
    %normalize the reference image        
    ref = ref./(max(max(ref)));
    %normalize the result image
    m = m./(max(max(m)));
    %IQA results 
    PSNR = psnr(m,ref);
    SSIM = ssim(m,ref);
    
end

function [M,m,error_image,weighted_Mpk,PSNR,SSIM] = homodyne(ratio_partial,ratio_symmetric)
    load('ge_phantom.mat');
    %reference image
    mf = ifft2c(kdata); 
    ref = abs(mf);
    
    %partial k-space data with the desired ratio
    [Mpk,mpk] = partial_kspace(kdata,ratio_partial);
    %the symmetric part of the k-space data
    [Ms,ms] = symmetric_kspace(kdata,ratio_symmetric);
    
    %multiplication of the partial k-space data with pre-weighting function
    weighted_Mpk = weightFunc(Mpk,ratio_symmetric);
    weighted_mpk = ifft2c(weighted_Mpk);    
    %phase corrected image
    phase_corrected_image = phase_correct(weighted_mpk,ms);
    
    %homodyne reconstruction image
    %take real part of the image as result image
    m = real(phase_corrected_image);  
    %k-space spectrum of the result image
    M = fft2c(m);
    %error image
    error_image = ref - m;   
    
    %normalize the reference image        
    ref = ref./(max(max(ref)));
    %normalize the result image
    m = m./(max(max(m)));
    %IQA results 
    PSNR = psnr(m,ref);
    SSIM = ssim(m,ref);
end 

function [M,m,error_image,PSNR,SSIM] = pocs(ratio_partial,ratio_symmetric,display_iter)
    load('ge_phantom.mat');
    %reference image
    mf = ifft2c(kdata);    
    ref = abs(mf);

    %partial k-space data with the desired ratio
    [Mpk,mpk] = partial_kspace(kdata,ratio_partial);
    %the symmetric part of the k-space data
    [Ms,ms] = symmetric_kspace(kdata,ratio_symmetric);
    %initial Mi(kx,ky) is Mpk(kx,ky)
    %later, the zeros will be filled with the phase constrained image data
    Mi = Mpk;
    
    for iter = 1:5
        if(iter >1)
           %data constrained k-space data  Mi(kx,ky)
           Mi = replace_kspace(Mpk,M,ratio_partial);
        end
        
        %phase of ms(x,y)
        p = exp(1i*angle(ms));         
        %phase constrained image data
        mi = ifft2c(Mi);
        m = abs(mi).*p;        
        %go back to k-space to use in the next iteration 
        M = fft2c(m);  
        
        if (display_iter)            
            figure;
            set(gcf, 'WindowState', 'maximized');
            %display the resulting k-space spectrum
            subplot(1,3,1);
            imshow(log(abs(M)+1),[]);
            title(strcat('k-space Spectrum for Iteration # ',num2str(iter)));
            xlabel('ky'); ylabel('kx');    

            %display the magnitude image
            subplot(1,3,2);
            imshow(abs(m),[]);
            title(strcat('Image for Iteration # ',num2str(iter)));
            xlabel('y'); ylabel('x');    

            %error image
            error_image = ref - real(mi.*exp(-j*angle(ms)));            
            %display the error image
            subplot(1,3,3);
            imshow(error_image,[]);
            title(strcat('Error Image for Iteration # ',num2str(iter)));
            xlabel('y'); ylabel('x');
            file = strcat('1.8_iteration',num2str(iter),'.png');
            saveas(gcf,file);            
            
            %normalize the reference image
            ref_IQA = ref./(max(max(ref)));
            %normalize the result image
            m_iter = real(mi.*exp(-1i*angle(ms)));
            m_iter = m_iter./(max(max(m_iter)));
            %IQA results 
            PSNR = psnr(m_iter,ref_IQA);
            SSIM = ssim(m_iter,ref_IQA);
            disp(strcat('PSNR #',num2str(iter),':',num2str(PSNR),...
                        ', SSIM #',num2str(iter),':',num2str(SSIM)));             
        end
    end
    
    %POCS reconstruction image
    %take real part of the image as result image
    m = real(mi.*exp(-j*angle(ms)));
    %k-space spectrum of the result image 
    M = fft2c(m);    
    %error image
    error_image = ref - m;
    
    %normalize the reference image        
    ref = ref./(max(max(ref)));
    %normalize the result image
    m = m./(max(max(m)));
    %IQA results 
    PSNR = psnr(m,ref);
    SSIM = ssim(m,ref);
    
end