function berfin_kavsut_21602459_hw2(question)
clc
close all

switch question
    
    %%part1 
    case '1.1'
        disp('1.1');
        disp('Spiral Trajectory');
        load('spiral.mat');
        %“interleaved spiral trajectory”
        % with 6 interleaves
        figure; set(gcf, 'WindowState', 'maximized'); plot(ktraj);
        xlabel('k_{x}'); ylabel('k_{y}');
        title('2D k-space Trajectory');
        %saveas(gcf,'1.1.png');
        
    case '1.2'
        disp('1.2');
        disp('Calculating Density Compensation Filter');
        load('spiral.mat');
        
        %density compensation filter = 1/density = area
        area = voronoidens(ktraj);        
        %density compensation filter until 2000th Data Point
        area_zoom = area(1:2000,:);       
        
        %number of NaN's in area 
        count_nan = sum(isnan(area));
        display(strcat('# of NaN for each trajectory: ', num2str(count_nan(1))));           
                        
        figure; set(gcf, 'WindowState', 'maximized');
        subplot(2,1,1); plot(area(:));
        xlabel('i^{th} Data Point'); ylabel('Area');
        title('Density Compensation Filter')
       
        subplot(2,1,2); plot(area_zoom(:));
        xlabel('i^{th} Data Point'); ylabel('Area');
        title('Density Compensation Filter until 2000^{th} Data Point');
        %saveas(gcf,'1.2.png');
        
    case '1.3'
        disp('1.3');
        disp('Correcting the Density Compensation Filter');
        load('spiral.mat');
                
        %NaN's and unreasonably large values are eliminated
        %by simple tresholding
        area = voronoidens(ktraj);
        corrected_dens = correct_dens(area);
        
        figure; set(gcf, 'WindowState', 'maximized');
        plot(corrected_dens(:));
        xlabel('i^{th} Data Point'); ylabel('Area');
        title('Corrected Density Compensation Filter');
        %saveas(gcf,'1.3.png');
        
    case '1.4'
       
        disp('1.4');
        disp('Direct Summation')
        
        load('spiral.mat');   
        %density compensation filter
        area = voronoidens(ktraj);
        d = correct_dens(area); 
        %for 128x128 image
        N = 128;      
        
        %normalized image calculated by direct summation technique
        %reference image for the rest of Part 1
        [ima_direct,cpu_time] = direct_summation(d,N,ktraj,kdata);
        %flip the image to have correct orientation 
        %on y-axis with imshow function 
        ima_direct = flipud(ima_direct);        

        %display the magnitude image
        figure; set(gcf, 'WindowState', 'maximized');
        imshow(ima_direct,[]);
        xlabel('x');ylabel('y');
        title('Direct Summation');
        %saveas(gcf,'1.4.png');
        
        %display the central horizontal cross-section of the image
        figure; set(gcf, 'WindowState', 'maximized');
        subplot(2,1,1);
        plot(ima_direct(end/2,:));
        xlabel('x'); ylabel('image');
        title('Central Horizontal Cross-Section');      
        
        %display the central vertical cross-section of this image
        subplot(2,1,2);
        plot(flip(ima_direct(:,end/2)));
        xlabel('y'); ylabel('image');
        title('Central Vertical Cross-Section');  
        %saveas(gcf,'1.4_central.png');
        
        disp(strcat('CPU time needed for direct summation technique: ', num2str(cpu_time)));
   
    case '1.5'

        disp('1.5');
        disp('Simple Gridding Without Density Compensation');

        load('spiral.mat');         
        %reference image 
        area = voronoidens(ktraj);
        d = correct_dens(area);
        N = 128;       
        [ref,time_directSum] = direct_summation(d,N,ktraj,kdata);
        ref = flipud(ref);

        %no density compensation
        t = cputime;
        w = ones(size(kdata));        
        ima = gridkb(kdata,ktraj,w,128,1,2,'image');
        time_simpleGrid = cputime-t;

        %correct orientation on y-axis for imshow function 
        image = flipud(ima);
        %normalize the image        
        image = abs(image);
        image = image./max(image(:));   

        %error image is difference of reference image 
        %and simple gridding image
        error = ref-image;       

        %display reference image      
        figure; set(gcf, 'WindowState', 'maximized');       
        subplot(1,3,1);
        imshow(ref,[]);
        xlabel('x'); ylabel('y');
        title('Reference Image');   

        %display result image
        subplot(1,3,2);
        imshow(image,[]);
        xlabel('x'); ylabel('y');
        title('Simple Gridding Without Density Compensation'); 

        %display magnitude of error image
        %to set the zero-error pixel to colour black 
        subplot(1,3,3);
        imshow(abs(error),[]);
        xlabel('x'); ylabel('y');
        title('Error Image');
        %saveas(gcf,'1.5.png');   

        %display the central horizontal cross-section of the image
        figure; set(gcf, 'WindowState', 'maximized');
        subplot(2,1,1);
        plot(ref(end/2,:),'m');
        hold on;  plot(image(end/2,:),'b');
        xlabel('x'); ylabel('image');
        title('Central Horizontal Cross-Section');          
        legend({'Reference Image','Simple Gridding Image'},'Location','NorthEast');

        %display the central vertical cross-section of the image
        subplot(2,1,2);
        plot(flip(ref(:,end/2)), 'm');
        hold on;  plot(flip(image(:,end/2)),'b');
        xlabel('y'); ylabel('image');
        title('Central Vertical Cross-Section'); 
        legend({'Reference Image','Simple Gridding Image'},'Location','NorthEast');

        %saveas(gcf,'1.5_central.png');   

        %IQA results 
        PSNR = psnr(image,ref);
        SSIM = ssim(image,ref);            
        disp(strcat('PSNR: ', num2str(PSNR)));
        disp(strcat('SSIM: ', num2str(SSIM))); 

        %CPU times time
        disp(strcat('CPU time needed for direct summation: ', num2str(time_directSum)));
        disp(strcat('CPU time needed for 1X gridding reconstruction: ', num2str(time_simpleGrid)));

    case '1.6'
        disp('1.6')
        disp('Simple Gridding With Density Compensation');
        
        load('spiral.mat'); 
        
        %reference image 
        area = voronoidens(ktraj);
        d = correct_dens(area);
        N = 128;       
        [ref,time_directSum] = direct_summation(d,N,ktraj,kdata);
        ref = flipud(ref);
        
        %simple gridding with density compensation
        t = cputime;
        w = d;        
        ima = gridkb(kdata,ktraj,w,128,1,2,'image');
        time_simpleGrid = cputime-t;
        
        %normalize the image 
        image = flipud(ima);
        image = abs(image);
        image = image./max(image(:));   

        %error image 
        error = ref-image; 
        
        %display reference image      
        figure; set(gcf, 'WindowState', 'maximized');       
        subplot(1,3,1);
        imshow(ref,[]);
        xlabel('x'); ylabel('y');
        title('Reference Image');   
        
        %display result image
        subplot(1,3,2);
        imshow(image,[]);
        xlabel('x'); ylabel('y');
        title('Simple Gridding With Density Compensation'); 
       
        %display error image
        subplot(1,3,3);
        imshow(abs(error),[]);
        xlabel('x'); ylabel('y');
        title('Error Image');
        %saveas(gcf,'1.6.png');   
        
        %display the central horizontal cross-section of the image
        figure; set(gcf, 'WindowState', 'maximized');
        subplot(2,1,1);
        plot(ref(end/2,:),'m');
        hold on;  plot(image(end/2,:),'b');
        xlabel('x'); ylabel('image');
        title('Central Horizontal Cross-Section');          
        legend({'Reference Image','Simple Gridding Image'},'Location','NorthEast');

        %display the central vertical cross-section of the image
        subplot(2,1,2);
        plot(flip(ref(:,end/2)), 'm');
        hold on;  plot(flip(image(:,end/2)),'b');
        xlabel('y'); ylabel('image');
        title('Central Vertical Cross-Section'); 
        legend({'Reference Image','Simple Gridding Image'},'Location','NorthEast');

        %saveas(gcf,'1.6_central.png');   

        %IQA results 
        PSNR = psnr(image,ref);
        SSIM = ssim(image,ref);            
        disp(strcat('PSNR: ', num2str(PSNR)));
        disp(strcat('SSIM: ', num2str(SSIM))); 

        %CPU times time
        disp(strcat('CPU time needed for direct summation: ', num2str(time_directSum)));
        disp(strcat('CPU time needed for 1X gridding reconstruction: ', num2str(time_simpleGrid))); 

    case '1.7'
       
        disp('1.7')
        disp('2X Gridding');
        
        load('spiral.mat');         
        %reference image 
        area = voronoidens(ktraj);
        d = correct_dens(area);
        N = 128;       
        [ref,time_directSum] = direct_summation(d,N,ktraj,kdata);
        ref = flipud(ref);
        
        %2X Gridding
        t = cputime;
        %density compensation
        w = d;        
        ima = gridkb(kdata,ktraj,w,128,2,4,'image');
        time_2Xgrid = cputime-t;

        %normalize full image
        full_image = flipud(ima);
        full_image = abs( full_image);
        full_image =  full_image./max( full_image(:));  

        %crop the image by taking central 128x128
        ima = ima(end/4+1:(3*end/4),end/4+1:(3*end/4));
        %normalize the image 
        image = flipud(ima);
        image = abs(image);
        image = image./max(image(:));   

        %error image
        error = ref - image;   

        %display full image 
        figure; set(gcf, 'WindowState', 'maximized'); 
        imshow(full_image,[]);
        xlabel('x'); ylabel('y');
        title('Full Image of 2X Gridding');  
        %saveas(gcf,'1.7_full_image.png'); 
        
        %display reference image      
        figure; set(gcf, 'WindowState', 'maximized');       
        subplot(1,3,1);
        imshow(ref,[]);
        xlabel('x'); ylabel('y');
        title('Reference Image');   
        
        %display result image
        subplot(1,3,2);
        imshow(image,[]);
        xlabel('x'); ylabel('y');
        title('2X Gridding'); 
       
        %display error image
        subplot(1,3,3);
        imshow(abs(error),[]);
        xlabel('x'); ylabel('y');
        title('Error Image');
        %saveas(gcf,'1.7.png');   
        
        %display the central horizontal cross-section of the image
        figure; set(gcf, 'WindowState', 'maximized');
        subplot(2,1,1);
        plot(ref(end/2,:),'m');
        hold on;  plot(image(end/2,:),'b');
        xlabel('x'); ylabel('image');
        title('Central Horizontal Cross-Section');          
        legend({'Reference Image','2X Gridding'},'Location','NorthEast');

        %display the central vertical cross-section of the image
        subplot(2,1,2);
        plot(flip(ref(:,end/2)), 'm');
        hold on;  plot(flip(image(:,end/2)),'b');
        xlabel('y'); ylabel('image');
        title('Central Vertical Cross-Section'); 
        legend({'Reference Image','2X Gridding'},'Location','NorthEast');
       
        %saveas(gcf,'1.7_central.png');   

        %IQA results        
        PSNR = psnr(image,ref);
        SSIM = ssim(image,ref);            
        disp(strcat('PSNR: ', num2str(PSNR)));
        disp(strcat('SSIM: ', num2str(SSIM))); 

        %CPU times time
        disp(strcat('CPU time needed for direct summation: ', num2str(time_directSum)));
        disp(strcat('CPU time needed for 2X gridding reconstruction: ', num2str(time_2Xgrid)));
        
    case '1.8'
        
        disp('1.8') 
        disp('Reduced Oversampling for Gridding');

        load('spiral.mat');         
        %reference image 
        area = voronoidens(ktraj);
        d = correct_dens(area);
        N = 128;       
        [ref,time_directSum] = direct_summation(d,N,ktraj,kdata);
        ref = flipud(ref);
        
        %reduced oversampling for gridding
        t = cputime;
        %density compensation
        w = d;       
        %for oversampling factor 1.25, I have chosen kernel width to be 6 
        %so that aliasing amplitude is less than 10^-3
        oversamp_factor = 1.25;
        kernel_width = 6;
        ima = gridkb(kdata,ktraj,w,N,oversamp_factor,kernel_width,'image');
        time_rog = cputime-t;

        %normalize full image
        full_image = flipud(ima);
        full_image = abs( full_image);
        full_image =  full_image./max( full_image(:));  

        %crop the image by taking central 128x128
        size_ima = 1.25*N;
        ima = ima(size_ima/2-N/2+1:size_ima/2+N/2,size_ima/2-N/2+1:size_ima/2+N/2);
        %%normalize the image 
        image = flipud(ima);
        image = abs(image);
        image = image./max(image(:));   

        %error image
        error = ref - image;   

        %display full image 
        figure; set(gcf, 'WindowState', 'maximized'); 
        imshow(full_image,[]);
        xlabel('x'); ylabel('y');
        title('Full Image of Gridding with Reduced Oversampling');  
        %saveas(gcf,'1.8_full_image.png'); 
        
        %display reference image      
        figure; set(gcf, 'WindowState', 'maximized');       
        subplot(1,3,1);
        imshow(ref,[]);
        xlabel('x'); ylabel('y');
        title('Reference Image');   
        
        %display result image
        subplot(1,3,2);
        imshow(image,[]);
        xlabel('x'); ylabel('y');
        title('Reduced Oversampling for Gridding'); 
       
        %display error image
        subplot(1,3,3);
        imshow(abs(error),[]);
        xlabel('x'); ylabel('y');
        title('Error Image');
        %saveas(gcf,'1.8.png');   
        
        %display the central horizontal cross-section of the image
        figure; set(gcf, 'WindowState', 'maximized');
        subplot(2,1,1);
        plot(ref(end/2,:),'m');
        hold on;  plot(image(end/2,:),'b');
        xlabel('x'); ylabel('image');
        title('Central Horizontal Cross-Section');          
        legend({'Reference Image','Reduced Oversampling for Gridding'},'Location','NorthEast');

        %display the central vertical cross-section of the image
        subplot(2,1,2);
        plot(flip(ref(:,end/2)), 'm');
        hold on;  plot(flip(image(:,end/2)),'b');
        xlabel('y'); ylabel('image');
        title('Central Vertical Cross-Section'); 
        legend({'Reference Image','Reduced Oversampling for Gridding'},'Location','NorthEast');
       
        %saveas(gcf,'1.8_central.png');   

        %IQA results        
        PSNR = psnr(image,ref);
        SSIM = ssim(image,ref);            
        disp(strcat('PSNR: ', num2str(PSNR)));
        disp(strcat('SSIM: ', num2str(SSIM))); 

        %CPU times time
        disp(strcat('CPU time needed for direct summation: ', num2str(time_directSum)));
        disp(strcat('CPU time needed for gridding reconstruction with oversampling ratio 1.25: ', num2str(time_rog)));
       
    case '1.9'
        
        disp('1.9')
        disp('Effect of Deapodization');
        
        load('spiral.mat');         
        %reference image 
        area = voronoidens(ktraj);
        d = correct_dens(area);
        N = 128;       
        [ref,time_directSum] = direct_summation(d,N,ktraj,kdata);
        ref = flipud(ref);

        %reduced oversampling for gridding
        t = cputime;
        %density compensation
        w = d;       
        %for oversampling factor 1.25, I have chosen kernel width to be 6 
        %so that aliasing amplitude is less than 10^-3
        oversamp_factor = 1.25;
        kernel_width = 5.5;
        %'k-space' is for eliminatig the effedct of deapodization
        kd = gridkb(kdata,ktraj,w,N,oversamp_factor,kernel_width,'k-space');
        ima = ifft2c(kd);
        time_rog = cputime-t;

        %normalize full image
        full_image = flipud(ima);
        full_image = abs( full_image);
        full_image =  full_image./max( full_image(:));  

        %crop the image by taking central 128x128
        size_ima = 1.25*N;
        ima = ima(size_ima/2-N/2+1:size_ima/2+N/2,size_ima/2-N/2+1:size_ima/2+N/2);
        %%normalize the image 
        image = flipud(ima);
        image = abs(image);
        image = image./max(image(:));   

        %error image
        error = ref - image;   

        %display full image 
        figure; set(gcf, 'WindowState', 'maximized'); 
        imshow(full_image,[]);
        xlabel('x'); ylabel('y');
        title('Full Image of Gridding without Deapodization');  
        %saveas(gcf,'1.9_full_image.png'); 
        
        %display reference image      
        figure; set(gcf, 'WindowState', 'maximized');       
        subplot(1,3,1);
        imshow(ref,[]);
        xlabel('x'); ylabel('y');
        title('Reference Image');   
        
        %display result image
        subplot(1,3,2);
        imshow(image,[]);
        xlabel('x'); ylabel('y');
        title('Gridding without Deapodization'); 
       
        %display error image
        subplot(1,3,3);
        imshow(abs(error),[]);
        xlabel('x'); ylabel('y');
        title('Error Image');
        %saveas(gcf,'1.9.png');   
        
        %display the central horizontal cross-section of the image
        figure; set(gcf, 'WindowState', 'maximized');
        subplot(2,1,1);
        plot(ref(end/2,:),'m');
        hold on;  plot(image(end/2,:),'b');
        xlabel('x'); ylabel('image');
        title('Central Horizontal Cross-Section');          
        legend({'Reference Image','Gridding without Deapodization'},'Location','NorthEast');

        %display the central vertical cross-section of the image
        subplot(2,1,2);
        plot(flip(ref(:,end/2)), 'm');
        hold on;  plot(flip(image(:,end/2)),'b');
        xlabel('y'); ylabel('image');
        title('Central Vertical Cross-Section'); 
        legend({'Reference Image','Gridding without Deapodization'},'Location','NorthEast');
       
        %saveas(gcf,'1.9_central.png');   

        %IQA results        
        PSNR = psnr(image,ref);
        SSIM = ssim(image,ref);            
        disp(strcat('PSNR: ', num2str(PSNR)));
        disp(strcat('SSIM: ', num2str(SSIM))); 

        %CPU times time
        disp(strcat('CPU time needed for direct summation: ', num2str(time_directSum)));
        disp(strcat('CPU time needed for gridding reconstruction without deapodization: ', num2str(time_rog)));
       
        
    %%part 2
    case '2.1'
        
        disp('2.1')
        disp('Generating Projection');
        
        P = phantom('Modified Shepp-Logan',256);
        proj = radon(P,[0:179]);

        %normalize phantom image to use as reference image
        ref = abs(P);
        ref = ref/max(ref(:));

        %take transpose to have L in x-axis 
        %and Theata in y-axis
        %then flip up-down to correct axis orientation         
        sinogram = flipud(transpose(proj));

        %display reference image          
        figure; set(gcf, 'WindowState', 'maximized');  
        subplot(1,2,1);
        imshow(abs(ref),[]);
        xlabel('x'); ylabel('y');
        title('Reference Image');   

        %display sinogram
        subplot(1,2,2);
        imshow(sinogram,[]);
        xlabel('{\it l}'); ylabel('\theta')
        title('Sinogram'); 
        %saveas(gcf,'2.1.png');        

        %display the central horizontal cross-section of the image
        figure; set(gcf, 'WindowState', 'maximized');
        subplot(2,1,1);
        plot(ref(end/2,:),'m');
        xlabel('x'); ylabel('image');
        title('Central Horizontal Cross-Section');          

        %display the central vertical cross-section of the image
        subplot(2,1,2);
        plot(flip(ref(:,end/2)), 'm');
        xlabel('y'); ylabel('image');
        title('Central Vertical Cross-Section'); 
        %saveas(gcf,'2.1_central.png');        
        
    case '2.2'
        disp('2.2');
        disp('Backprojection');
        
        %reference image
        P = phantom('Modified Shepp-Logan',256);
        proj = radon(P,[0:179]);        
        ref = abs(P);
        ref = ref/max(ref(:));

        %filtered backprojection    
        image = iradon(proj,0:179);        
        image = image(2:end-1,2:end-1);
        %normalize the image
        image = abs(image);
        image = image/max(image(:));

        %error image 
        error = ref - image;

        %display reference image      
        figure; set(gcf, 'WindowState', 'maximized');       
        subplot(1,3,1);
        imshow(ref,[]);
        xlabel('x'); ylabel('y');
        title('Reference Image');   

        %display result image
        subplot(1,3,2);
        imshow(image,[]);
        xlabel('x'); ylabel('y');
        title('Filtered Backprojection'); 

        %display magnitude of error image
        %to set the zero-error pixel to colour black 
        subplot(1,3,3);
        imshow(abs(error),[]);
        xlabel('x'); ylabel('y');
        title('Error Image');
        %saveas(gcf,'2.2.png');   

        %display central cross-sections of the image
        %compare reference image cross-sections with filtered backprojection
        figure; set(gcf, 'WindowState', 'maximized');
        subplot(2,1,1);
        plot(ref(end/2,:),'m');
        hold on;  plot(image(end/2,:),'b');
        xlabel('x'); ylabel('image');
        title('Central Horizontal Cross-Section');          
        legend({'Reference Image','Filtered Backprojection'},'Location','NorthEast');

        subplot(2,1,2);
        plot(flip(ref(:,end/2)), 'm');
        hold on;  plot(flip(image(:,end/2)),'b');
        xlabel('y'); ylabel('image');
        title('Central Vertical Cross-Section'); 
        legend({'Reference Image','Filtered Backprojection'},'Location','NorthEast');

        %saveas(gcf,'2.2_central.png');   

        %IQA results 
        PSNR = psnr(image,ref);
        SSIM = ssim(image,ref);            
        disp(strcat('PSNR: ', num2str(PSNR)));
        disp(strcat('SSIM: ', num2str(SSIM))); 

    case '2.3'
        
        disp('2.3');
        disp('Naive Backprojection');
        
        %reference image
        P = phantom('Modified Shepp-Logan',256);
        proj = radon(P,[0:179]);        
        ref = abs(P);
        ref = ref/max(ref(:));

        %naive backprojection    
        image = iradon(proj,0:179,'none');        
        image = image(2:end-1,2:end-1);
        %normalize the image
        image = abs(image);
        image = image/max(image(:));

        %error image 
        error = ref - image;

        %display reference image      
        figure; set(gcf, 'WindowState', 'maximized');       
        subplot(1,3,1);
        imshow(ref,[]);
        xlabel('x'); ylabel('y');
        title('Reference Image');   

        %display result image
        subplot(1,3,2);
        imshow(image,[]);
        xlabel('x'); ylabel('y');
        title('Naive Backprojection'); 

        %display magnitude of error image
        %to set the zero-error pixel to colour black 
        subplot(1,3,3);
        imshow(abs(error),[]);
        xlabel('x'); ylabel('y');
        title('Error Image');
        %saveas(gcf,'2.3.png');   

        %display central cross-sections of the image
        %compare reference image cross-sections with filtered backprojection
        figure; set(gcf, 'WindowState', 'maximized');
        subplot(2,1,1);
        plot(ref(end/2,:),'m');
        hold on;  plot(image(end/2,:),'b');
        xlabel('x'); ylabel('image');
        title('Central Horizontal Cross-Section');          
        legend({'Reference Image','Naive Backprojection'},'Location','NorthEast');

        subplot(2,1,2);
        plot(flip(ref(:,end/2)), 'm');
        hold on;  plot(flip(image(:,end/2)),'b');
        xlabel('y'); ylabel('image');
        title('Central Vertical Cross-Section'); 
        legend({'Reference Image','Naive Backprojection'},'Location','NorthEast');

        %saveas(gcf,'2.3_central.png');   

        %IQA results 
        PSNR = psnr(image,ref);
        SSIM = ssim(image,ref);            
        disp(strcat('PSNR: ', num2str(PSNR)));
        disp(strcat('SSIM: ', num2str(SSIM))); 

    case '2.4'
        
        disp('2.4')
        disp('Generating k-space Data from Projections')
        
        %projection slice theorem 
        P = phantom('Modified Shepp-Logan',256);
        proj = radon(P,[0:179]);        
        [ktraj,kspace] = projection_slice(proj);
        kspace = flipud(transpose(kspace));

        %180 radial line in k-space
        %theta is between [0,179] 
        %radius is between [-0.5,0.5]
        figure; set(gcf, 'WindowState', 'maximized'); 
        subplot(1,2,1); imshow(log(abs(kspace)+1),[]);
        title('K-space Data');
        xlabel('k_{\rho}'); ylabel('\theta');

        subplot(1,2,2); plot(ktraj);
        xlabel('k_{x}'); ylabel('k_{y}');
        title('Radial Trajectory');
        %saveas(gcf,'2.4.png');
        
    case '2.5'
        disp('2.5');
        disp('Density Compensation Filter for Radial Trajectory');
        
        P = phantom('Modified Shepp-Logan',256);
        proj = radon(P,[0:179]);       

        %density compensation filter calculated by geometric approach 
        [filter,rho] = density_compensate(proj,'opt1');
        filter_first = filter(:,1);

        %display the density compensation filter for radila trajectory 
        figure; set(gcf, 'WindowState', 'maximized');
        plot(rho,filter_first);
        xlabel('k_{\rho}'); ylabel('Area');
        title('Density Compensation Filter for Radial Trajectory');
        %saveas(gcf,'2.5.png');    
        
    case '2.6'
        disp('2.6')
        disp('Direct Summation for Radial Trajectory')
        
        %reference image
        P = phantom('Modified Shepp-Logan',256);
        proj = radon(P,[0:179]);  
        ref = abs(P);
        ref = ref/max(ref(:));

        %density compensation filter 
        [filter,rho] = density_compensate(proj,'opt1');
        %projection slice theorem
        [ktraj,kspace] = projection_slice(proj);
        %filter
        d = filter;
        %256x256 image
        N = 256;
        %direct summation 
        [ima_direct,time] = direct_summation(d,N,ktraj,kspace);

        %normalize the result image
        image = abs(ima_direct);
        image = image/max(image(:));  
        image = flipud(transpose(image));

        %error image
        error = ref - image;

        %display the reference image
        figure; set(gcf, 'WindowState', 'maximized');
        subplot(1,3,1);imshow(ref,[]);
        xlabel('x');ylabel('y');
        title('Reference Image');
        
        subplot(1,3,2);imshow(image,[]);
        xlabel('x');ylabel('y');
        title('Direct Summation for Radial Trajectory');
              
        subplot(1,3,3);imshow(abs(error),[]);
        xlabel('x');ylabel('y');
        title('Error Image');
        %saveas(gcf,'2.6.png');
        
        %display central cross-sections of the image
        %compare the reference image cross-sections with direct summation 
        figure; set(gcf, 'WindowState', 'maximized');
        subplot(2,1,1);
        plot(ref(end/2,:),'m');
        hold on;  plot(image(end/2,:),'b');
        xlabel('x'); ylabel('image');
        title('Central Horizontal Cross-Section');          
        legend({'Reference Image','Direct Summation'},'Location','NorthEast');

        subplot(2,1,2);
        plot(flip(ref(:,end/2)), 'm');
        hold on;  plot(flip(image(:,end/2)),'b');
        xlabel('y'); ylabel('image');
        title('Central Vertical Cross-Section'); 
        legend({'Reference Image','Direct Summation'},'Location','NorthEast');

        %saveas(gcf,'2.6_central.png');   
        
        %IQA results 
        PSNR = psnr(image,ref);
        SSIM = ssim(image,ref);            
        disp(strcat('PSNR: ', num2str(PSNR)));
        disp(strcat('SSIM: ', num2str(SSIM))); 
    
    case '2.7'
        disp('2.7')
        disp('1X Gridding Reconstruction')     
        
        %reference image
        P = phantom('Modified Shepp-Logan',256);
        proj = radon(P,[0:179]);  
        ref = abs(P);
        ref = ref/max(ref(:));

        %density compensation filter 
        [filter,rho] = density_compensate(proj,'opt1');
        %projection slice theorem
        [ktraj,kdata] = projection_slice(proj);
        %filter
        w = filter;
        %256x256 image
        N = 256;        
        %1X gridding reconstruction 
        ima_grid = gridkb(kdata,ktraj,w,256,1,2,'image');

        %normalize the result image
        image = abs(ima_grid);
        image = image/max(image(:));
        image = flipud(transpose(image));

        %error image
        error = ref - image;

        %display the reference image
        figure; set(gcf, 'WindowState', 'maximized');
        subplot(1,3,1);imshow(ref,[]);
        xlabel('x');ylabel('y');
        title('Reference Image');
        
        subplot(1,3,2);imshow(image,[]);
        xlabel('x');ylabel('y');
        title('1X Gridding');
              
        subplot(1,3,3);imshow(abs(error),[]);
        xlabel('x');ylabel('y');
        title('Error Image');
        %saveas(gcf,'2.7.png');
        
        %display central cross-sections of the image
        %compare the reference image cross-sections with 1x gridding
        figure; set(gcf, 'WindowState', 'maximized');
        subplot(2,1,1);
        plot(ref(end/2,:),'m');
        hold on;  plot(image(end/2,:),'b');
        xlabel('x'); ylabel('image');
        title('Central Horizontal Cross-Section');          
        legend({'Reference Image','1X Gridding'},'Location','NorthEast');

        subplot(2,1,2);
        plot(flip(ref(:,end/2)), 'm');
        hold on;  plot(flip(image(:,end/2)),'b');
        xlabel('y'); ylabel('image');
        title('Central Vertical Cross-Section'); 
        legend({'Reference Image','1X Gridding'},'Location','NorthEast');

        %saveas(gcf,'2.7_central.png');   
        
        %IQA results 
        PSNR = psnr(image,ref);
        SSIM = ssim(image,ref);            
        disp(strcat('PSNR: ', num2str(PSNR)));
        disp(strcat('SSIM: ', num2str(SSIM))); 
        
    case '2.8'
        disp('2.8')
        disp('2X Gridding Reconstruction')
        
        %reference image
        P = phantom('Modified Shepp-Logan',256);
        proj = radon(P,[0:179]);  
        ref = abs(P);
        ref = ref/max(ref(:));

        %density compensation filter 
        [filter,rho] = density_compensate(proj,'opt1');
        %filter
        w = filter;
        %projection slice theorem
        [ktraj,kdata] = projection_slice(proj);

        %2X gridding reconstruction 
        ima_grid = gridkb(kdata,ktraj,w,256,2,4,'image');
        %crop the image by taking central 128x128
        ima_grid = ima_grid(end/4+1:(3*end/4),end/4+1:(3*end/4));  
        %normalize the result image
        image = abs(ima_grid);
        image = image/max(image(:));
        image = flipud(transpose(image));

        %error image
        error = ref - image;
        
        %display the reference image
        figure; set(gcf, 'WindowState', 'maximized');
        subplot(1,3,1);imshow(ref,[]);
        xlabel('x');ylabel('y');
        title('Reference Image');
        
        subplot(1,3,2);imshow(image,[]);
        xlabel('x');ylabel('y');
        title('2X Gridding');
              
        subplot(1,3,3);imshow(abs(error),[]);
        xlabel('x');ylabel('y');
        title('Error Image');
        %saveas(gcf,'2.8.png');
        
        %display central cross-sections of image 
        %with comparison to reference image
        figure; set(gcf, 'WindowState', 'maximized');
        subplot(2,1,1);
        plot(ref(end/2,:),'m');
        hold on;  plot(image(end/2,:),'b');
        xlabel('x'); ylabel('image');
        title('Central Horizontal Cross-Section');          
        legend({'Reference Image','2X Gridding'},'Location','NorthEast');

        subplot(2,1,2);
        plot(flip(ref(:,end/2)), 'm');
        hold on;  plot(flip(image(:,end/2)),'b');
        xlabel('y'); ylabel('image');
        title('Central Vertical Cross-Section'); 
        legend({'Reference Image','2X Gridding'},'Location','NorthEast');

        %saveas(gcf,'2.8_central.png');   
        
        %IQA results 
        ref_IQA = abs(ref);
        ima_IQA = abs(image);
        PSNR = psnr(ima_IQA,ref_IQA);
        SSIM = ssim(ima_IQA,ref_IQA);            
        disp(strcat('PSNR: ', num2str(PSNR)));
        disp(strcat('SSIM: ', num2str(SSIM))); 
        
    %%part3
    case '3.1'
        disp('3.1')
       
        %ktraj:129x300 matrix, kdata:129x300 matrix
        load('shepplogan_radial_data.mat');

        figure; set(gcf, 'WindowState', 'maximized'); plot(ktraj);
        xlabel('k_{x}'); ylabel('k_{y}');
        title('2D k-space Trajectory');
        %saveas(gcf,'3.1.png');
  
    case '3.2'
        disp('3.2');
        disp('Density Compensation Filter for Radial Trajectory');
        
        load('shepplogan_radial_data.mat');
        %density compensation filter calculated by geometric approach 
        [filter,rho] = density_compensate(ktraj,'opt2');
        filter_first = filter(:,1);

        %display the density compensation filter for radila trajectory 
        figure; set(gcf, 'WindowState', 'maximized');
        plot(rho,filter_first);
        xlabel('k_{\rho}'); ylabel('Area');
        title('Density Compensation Filter for Radial Trajectory');
        %saveas(gcf,'3.2.png');
        
    case '3.3'
       
        disp('3.3')
        disp('Direct Summation for Radial Trajectory')
        
        %reference image
        P = phantom('Modified Shepp-Logan',256);
        ref = abs(P);
        ref = ref/max(ref(:));

        load('shepplogan_radial_data.mat');
        %density compensation filter 
        [filter,rho] = density_compensate(ktraj,'opt2');
        %256x256 image
        N = 256;
        %filter
        d = filter;
        %direct summation reconstruction 
        [ima_direct,time] = direct_summation(d,N,ktraj,kdata);

        %normalize the result image
        image = abs(ima_direct);
        image = image/max(image(:));
        image = flipud(transpose(image));

        %error image
        error = ref-image;
       
         %display the reference image
        figure; set(gcf, 'WindowState', 'maximized');
        subplot(1,3,1);imshow(ref,[]);
        xlabel('x');ylabel('y');
        title('Reference Image');
        
        subplot(1,3,2);imshow(image,[]);
        xlabel('x');ylabel('y');
        title('Direct Summation');
              
        subplot(1,3,3);imshow(abs(error),[]);
        xlabel('x');ylabel('y');
        title('Error Image');
        %saveas(gcf,'3.3.png');
        
        %display central cross-sections of image 
        %with comparison to reference image
        figure; set(gcf, 'WindowState', 'maximized');
        subplot(2,1,1);
        plot(ref(end/2,:),'m');
        hold on;  plot(image(end/2,:),'b');
        xlabel('x'); ylabel('image');
        title('Central Horizontal Cross-Section');          
        legend({'Reference Image','Direct Summation'},'Location','NorthEast');

        subplot(2,1,2);
        plot(flip(ref(:,end/2)), 'm');
        hold on;  plot(flip(image(:,end/2)),'b');
        xlabel('y'); ylabel('image');
        title('Central Vertical Cross-Section'); 
        legend({'Reference Image','Direct Summation'},'Location','NorthEast');

        %saveas(gcf,'3.3_central.png');   
        
        %IQA results 
        PSNR = psnr(image,ref);
        SSIM = ssim(image,ref);            
        disp(strcat('PSNR: ', num2str(PSNR)));
        disp(strcat('SSIM: ', num2str(SSIM))); 
        
    case '3.4'
        disp('3.4')
        disp('1X Gridding Reconstruction')

        %reference image
        P = phantom('Modified Shepp-Logan',256);
        ref = abs(P);
        ref = ref/max(ref(:));

        load('shepplogan_radial_data.mat');
        %density compensation filter 
        [filter,rho] = density_compensate(ktraj,'opt2');
        %filter
        w = filter;
        %1X gridding reconstruction 
        ima_grid = gridkb(kdata,ktraj,w,256,1,2,'image');

        %normalize the result image
        image = abs(ima_grid);
        image = image/max(image(:));
        image = flipud(transpose(image));

        %error image
        error = ref - image;

         %display the reference image
        figure; set(gcf, 'WindowState', 'maximized');
        subplot(1,3,1);imshow(ref,[]);
        xlabel('x');ylabel('y');
        title('Reference Image');
        
        subplot(1,3,2);imshow(image,[]);
        xlabel('x');ylabel('y');
        title('1X Gridding');
              
        subplot(1,3,3);imshow(abs(error),[]);
        xlabel('x');ylabel('y');
        title('Error Image');
        %saveas(gcf,'3.4.png');
        
        %display central cross-sections of image 
        %with comparison to reference image
        figure; set(gcf, 'WindowState', 'maximized');
        subplot(2,1,1);
        plot(ref(end/2,:),'m');
        hold on;  plot(image(end/2,:),'b');
        xlabel('x'); ylabel('image');
        title('Central Horizontal Cross-Section');          
        legend({'Reference Image','1X Gridding'},'Location','NorthEast');

        subplot(2,1,2);
        plot(flip(ref(:,end/2)), 'm');
        hold on;  plot(flip(image(:,end/2)),'b');
        xlabel('y'); ylabel('image');
        title('Central Vertical Cross-Section'); 
        legend({'Reference Image','1X Gridding'},'Location','NorthEast');

        %saveas(gcf,'3.4_central.png');   
        
        %IQA results 
        ref_IQA = abs(ref);
        ima_IQA = abs(image);
        PSNR = psnr(ima_IQA,ref_IQA);
        SSIM = ssim(ima_IQA,ref_IQA);            
        disp(strcat('PSNR: ', num2str(PSNR)));
        disp(strcat('SSIM: ', num2str(SSIM))); 
       
        
     case '3.5'
        disp('3.5')
        disp('2X Gridding Reconstruction')
        
        %reference image
        P = phantom('Modified Shepp-Logan',256);
        ref = abs(P);
        ref = ref/max(ref(:));

        load('shepplogan_radial_data.mat');
        %density compensation filter 
        [filter,rho] = density_compensate(ktraj,'opt2');
        %filter
        w = filter;
        %2X gridding reconstruction 
        ima_grid = gridkb(kdata,ktraj,w,256,2,4,'image');
        %crop the image by taking central 128x128
        ima_grid = ima_grid(end/4+1:(3*end/4),end/4+1:(3*end/4));  
        %normalize the result image
        image = abs(ima_grid);
        image = image/max(image(:));
        image = flipud(transpose(image));

        %error image
        error = ref - image;
        
        %display the reference image
        figure; set(gcf, 'WindowState', 'maximized');
        subplot(1,3,1);imshow(ref,[]);
        xlabel('x');ylabel('y');
        title('Reference Image');
        
        subplot(1,3,2);imshow(image,[]);
        xlabel('x');ylabel('y');
        title('2X Gridding');
              
        subplot(1,3,3);imshow(abs(error),[]);
        xlabel('x');ylabel('y');
        title('Error Image');
        %saveas(gcf,'3.5.png');
        
        %display central cross-sections of image 
        %with comparison to reference image
        figure; set(gcf, 'WindowState', 'maximized');
        subplot(2,1,1);
        plot(ref(end/2,:),'m');
        hold on;  plot(image(end/2,:),'b');
        xlabel('x'); ylabel('image');
        title('Central Horizontal Cross-Section');          
        legend({'Reference Image','2X Gridding'},'Location','NorthEast');

        subplot(2,1,2);
        plot(flip(ref(:,end/2)), 'm');
        hold on;  plot(flip(image(:,end/2)),'b');
        xlabel('y'); ylabel('image');
        title('Central Vertical Cross-Section'); 
        legend({'Reference Image','2X Gridding'},'Location','NorthEast');

        %saveas(gcf,'3.5_central.png');   
        
        %IQA results 
        ref_IQA = abs(ref);
        ima_IQA = abs(image);
        PSNR = psnr(ima_IQA,ref_IQA);
        SSIM = ssim(ima_IQA,ref_IQA);            
        disp(strcat('PSNR: ', num2str(PSNR)));
        disp(strcat('SSIM: ', num2str(SSIM))); 
      
        
end

end


function d = fft1c(f)
% d = fft21(f)
%
% fft1c performs a centered fft
d = fftshift(fft(ifftshift(f)));
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


function corrected = correct_dens(area)   
%
% function corrected = correct_dens(area);
%
% input:  area calculated for density compensation filter
% output: density compensation filter without NaN's 
%         and unreasonably large values

[m,n] = size(area);
corrected = 0.0001*ones(size(area));
nan_check = isnan(area);
for m = 1:m
    for n = 1:n
        if (nan_check(m,n) == 0)
            corrected(m,n) = area(m,n);
        end
    end
end
corrected(corrected >= 0.0001) = 0.0001;        
end

function [image,time] = direct_summation(d,N,ktraj,kdata)
%
% function [ima_direct,time] = direct_summation(d,N,ktraj,kdata)
% 
% input:  d = density compensation filter
%         N = #of pixels, image has dimensions of NxN
%         ktraj = kx + i ky, k-space trajectory
%         kdata = k-space data 
% output: ima_direct = normalized magnitude image

%start the timer
t=cputime;
 
%(kx,ky) coordinates
kx = real(ktraj);
ky = imag(ktraj);

%tau is the center of field-of-view relative to the image
%for going from -64 to +64 in x-y coordinates
tau =  N/2;    

%direct summation technique
ima_direct = zeros(N,N);
for m = 0:N-1
    for  n = 0:N-1              
      ima_direct(m+1,n+1) = sum(sum(d.*kdata.*exp(1i*2*pi*(kx.*(m-tau) + ky.*(n-tau)))));     
    end
end

%take magnitude image 
image = abs(ima_direct);
%normalize the magnitude image
image = image/max(image(:)); 

time = cputime-t;

end

function [ktraj,kspace] = projection_slice(proj)
%
% function [ktraj,kspace] = projection_slice(proj)
% 
% input:  proj = g(l,theta) 
% output: ktraj = kx + j*ky
%         kspace = G(l,theta)

%line_no is equal to number of angles we are projecting with
%datapoint_no is number of values sampled for each projection 
[datapoint_no,line_no] = size(proj);

%radius is from -0.5 to 0.5
radius = linspace(-0.5,0.5,datapoint_no);
%theta is from 0 to 180
%convert degrees to radians 
theta = [0:line_no]/180*pi;

%ktraj: (kx,ky) = (r*cos(theta),r*sin(theta)) 
ktraj = zeros(datapoint_no,line_no);
for i = 1:180
    ktraj(:,i) = radius*exp(j*theta(i));
end 

%take 1D FFT of proj for k-space 
%fft() is applied to each column,
%which means we take FFT of each projection  
kspace = fft1c(proj);

end

function [filter,rho] = density_compensate(proj,opt)
%
% function [filter,rho] = density_compensate(proj,opt)
% 
% input:  proj = g(l,theta) 
%         opt =   'opt1', 'opt2'
% output: filter = calculated area for each sample point
%         rho = k-rho vector for given projectory 

%'opt1' : radial lines extend from -0.5 to 0.5 
%'opt2' : radial extend from the center, i.e. from 0 to 0.5  
switch opt

case 'opt1'

[datapoint_no,line_no] = size(proj);

%N is number of lines
N = line_no;

%k is between -0.5 and 0.5 
k_interval = 1/(datapoint_no-1);

%each sample is in n^th disk 
limit = (datapoint_no - 1)/2;
rho = [-limit:limit];
n = abs(rho);

%calculate area for each sample 
area = zeros(1,datapoint_no);
%n^th sample 
area = (pi/N)*(k_interval)^2.*n; 
%DC disk
area(rho == 0) = (pi/(4*N))*(k_interval/2)^2;

%create filter for each data point
%columns are  equal, because filter is equal each projection 
filter = transpose(repmat(area,[line_no 1]));

case 'opt2' 
    
[datapoint_no,line_no] = size(proj);  

%we treat lines as if 2 opposite lines are combined to 1 line 
N = line_no/2;  

%k is between 0 and 0.5 
k_interval = 0.5/datapoint_no;

%each sample is in n^th disk 
limit = datapoint_no-1;
rho = [0:limit];
n = rho;

%calculate area for each sample 
area = zeros(1,datapoint_no);
%n^th sample 
area = (pi/N)*(k_interval)^2.*n; 
%DC disk
area(n == 0) = (pi/(4*N))*(k_interval/2)^2;

%create filter for each data point
%columns are  equal, because filter is equal each projection 
filter = transpose(repmat(area,[line_no 1]));

end 

end


