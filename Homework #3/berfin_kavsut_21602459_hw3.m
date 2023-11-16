function berfin_kavsut_21602459_hw3(question)
%clc
%close all

switch question

case '1.1'

disp('1.1');
disp('Preparing System Matrix');

load("SM_40x40.mat");

%For both coils, plot the magnitude spectrum for
%calibration measurements at three different grid locations
figure;set(gcf, 'WindowState', 'maximized');
subplot(3,1,1); plot(abs(fftc(SM_coil1(:,1))));
title('Coil #1 1^{st} Grid Location');
xlabel('Row Number'); ylabel('Frequency Component');
subplot(3,1,2); plot(abs(fftc(SM_coil1(:,10))));  
title('Coil #1 10^{th} Grid Location');
xlabel('Row Number'); ylabel('Frequency Component');
subplot(3,1,3); plot(abs(fftc(SM_coil1(:,100))));
title('Coil #1 100^{th} Grid Location');
xlabel('Row Number'); ylabel('Frequency Component');
saveas(gcf,'1.1_Coil#1.png');

figure;set(gcf, 'WindowState', 'maximized');
subplot(3,1,1); plot(abs(fftc(SM_coil2(:,1))));
title('Coil #2 1^{st} Grid Location');
xlabel('Row Number'); ylabel('Frequency Component');
subplot(3,1,2); plot(abs(fftc(SM_coil2(:,10))));  
title('Coil #2 10^{th} Grid Location');
xlabel('Row Number'); ylabel('Frequency Component');
subplot(3,1,3); plot(abs(fftc(SM_coil2(:,100))));
title('Coil #2 100^{th} Grid Location');
xlabel('Row Number'); ylabel('Frequency Component');
saveas(gcf,'1.1_Coil#2.png');

%prepare the system matrix 

%FFT from calibration measurement from Coil #1
S1 = zeros(size(SM_coil1));
for n = 1:1600
    %size: 10001x1 for each column vector
    S1(:,n) = fftc(SM_coil1(:,n));
end
%size: 5001x1600
S1(5001:10001,:) = [];
%size: 10002x1600
S1 = [real(S1);imag(S1)];

%FFT from calibration measurement from Coil #1
S2 = zeros(size(SM_coil2));
for n = 1:1600
    %size: 10001x1 for each column vector
    S2(:,n) = fftc(SM_coil2(:,n));
end
%size: 5001x1600
S2(5001:10001,:) = [];
%size: 10002x1600
S2 = [real(S2);imag(S2)];

%size of system matrix: 20004x1600
S = [S1;S2];

%Plot three different columns of the system matrix
figure;set(gcf, 'WindowState', 'maximized');
subplot(3,1,1); plot( S(:,1) );
title('1^{st} Grid Location');
xlabel('Row Number'); ylabel('Frequency Component');
subplot(3,1,2); plot( S(:,10) );
title('10^{th} Grid Location');
 xlabel('Row Number'); ylabel('Frequency Component');
subplot(3,1,3); plot( S(:,100) );
title('100^{th} Grid Location');
 xlabel('Row Number'); ylabel('Frequency Component');
saveas(gcf,'1.1_System_Matrix.png');

%save S, system matrix, as a mat-file 
save("S.mat",'S');

case '1.2'

disp('1.2');
disp("Preparing Measurement Vector");

load("measurement_phantom.mat",'meas_coil1','meas_coil2','phantom');

%display phantom, which will be reference image
figure;set(gcf, 'WindowState', 'maximized');
imshow(phantom,[]); title('Reference Image');
saveas(gcf,'1.2_phantom.png');

%prepare measurement vector 

%measurement from coil #1
%size: 10001x1
u1 = fftc(meas_coil1);
%size: 5001x1
u1(5001:10001) = [];
%size: 10002x1
u1 = [real(u1);imag(u1)];

%measurement from coil #2
%size: 10001x1
u2 = fftc(meas_coil2);
%size: 5001x1
u2(5001:10001) = [];
%size: 10002x1
u2 = [real(u2);imag(u2)];

%20004x1
u = [u1;u2];

figure;set(gcf, 'WindowState', 'maximized');
plot(u); title('Measurement Vector');
xlabel('Row Number'); ylabel('Frequency Component');
saveas(gcf,'1.2_measurement_vector.png');

%save u, measurement vector, as a mat-file
save("u.mat",'u');

case '1.3'

disp('1.3');
disp("SVD");

load("S.mat");

[U,Sigma,V] = svd(S,'econ');
%U: size of 20004x1600
%Sigma: diagonal matrix, size of 1600x1600    
%V: size of 1600x1600
size_U = size(U)
size_Sigma = size(Sigma)
size_V = size(V)

%plot the singular values (i.e. diagonal entries of Sigma)
singular_values = diag(Sigma);
figure;set(gcf, 'WindowState', 'maximized');
plot(singular_values); title('Singular Values');
xlabel('i'); ylabel('{\sigma}_i');
saveas(gcf,'1.3_singular_values.png');   

condition_no = singular_values(1)/singular_values(end);
condition_no_matlab = cond(Sigma) 
%gives the same result of 725.6196
disp(strcat('Condition Number: ', num2str(condition_no))); 

case '1.4'

disp('1.4');
disp("Reconstruction via SVD");    

load("S.mat"); load("u.mat");
load("measurement_phantom.mat",'meas_coil1','meas_coil2','phantom');

%SVD of System Matrix 
[U,Sigma,V] = svd(S,'econ');
%compute the image c using Moore-Pensore 
%pseudo inverse from SVD    
c = V*inv(Sigma)*U'*u; %conjugate transpose of U = U'
%alternative solution: c = V*((U'*u)./diag(Sigma));

%construct the image
ima = reshape(c,40,40);
%resize the image for 280x280
ima = imresize(ima, [280 280]);
%set negative values to zero
ima(ima<0) = 0; 

figure;set(gcf, 'WindowState', 'maximized');
subplot(1,3,1); imshow(phantom,[]); 
title('Reference Image');
subplot(1,3,2); imshow(ima,[]); 
title('Reconstruction Image via SVD'); 
subplot(1,3,3); imshow(abs(phantom-ima),[]); 
title('Error Image'); 
saveas(gcf,'1.4.png');

%IQA measurements
PSNR = psnr(ima,phantom);
SSIM = ssim(ima,phantom);            
disp(strcat('PSNR: ', num2str(PSNR)));
disp(strcat('SSIM: ', num2str(SSIM))); 

%comment on image

case '1.5'

disp('1.5');
disp("Truncated SVD");    

load("S.mat");load("u.mat");
load("measurement_phantom.mat",'meas_coil1','meas_coil2','phantom');

[U,Sigma,V] = svd(S,'econ');
singular_values = diag(Sigma);

condition_no = 0;
for i = 1: length(singular_values)
    condition_no = singular_values(1)/singular_values(i);
    if(condition_no >=100)
        N = i;
        break;
    end
end

disp(strcat('Condition Number: ', num2str(condition_no)));
disp(strcat('N: ', num2str(N)));

U(:,(N+1):end) = []; %U: 20004xN
V(:,(N+1):end) = []; %V: 1600xN
Sigma = Sigma(1:N,1:N); %Sigma: NxN  

c = V*inv(Sigma)*U'*u;

ima = reshape(c,40,40);
ima = imresize(ima, [280 280]);
ima(ima<0) = 0; 

figure;set(gcf, 'WindowState', 'maximized');
subplot(1,3,1); imshow(phantom,[]); 
title('Reference Image');
subplot(1,3,2); imshow(ima,[]); 
title('Truncated SVD'); 
subplot(1,3,3); imshow(abs(phantom-ima),[]); 
title('Error Image'); 
saveas(gcf,'1.5.png');

%IQA measurements
PSNR = psnr(ima,phantom);
SSIM = ssim(ima,phantom);            
disp(strcat('PSNR: ', num2str(PSNR)));
disp(strcat('SSIM: ', num2str(SSIM))); 

%comment on improvements 

case '1.6'

disp('1.6');
disp("Truncated SVD with Different Condition Numbers");

load("S.mat"); load("u.mat");
load("measurement_phantom.mat",'meas_coil1','meas_coil2','phantom');

[U_ref,Sigma_ref,V_ref] = svd(S,'econ');
singular_values = diag(Sigma_ref);

cond = [5, 10, 20, 30, 50]; 
for k = 1:length(cond)      
    condition_no = 0;
    for i = 1: length(singular_values)
        condition_no = singular_values(1)/singular_values(i);
        if(condition_no >= cond(k))
            N = i;
            break;
        end
    end

    disp(strcat('Condition Number: ', num2str(condition_no)));
    disp(strcat('N: ', num2str(N)));

    U = U_ref(:,1:N); %U: 20004xN
    V = V_ref(:,1:N); %V: 1600xN
    Sigma = Sigma_ref(1:N,1:N); %Sigma: NxN  

    c = V*inv(Sigma)*U'*u;

    ima = reshape(c,40,40);
    ima = imresize(ima, [280 280]);
    ima(ima<0) = 0; 

    figure;set(gcf, 'WindowState', 'maximized');
    subplot(1,3,1); imshow(phantom,[]); 
    title('Reference Image');
    subplot(1,3,2); imshow(ima,[]); 
    title(strcat('Truncated SVD (Condition Number = ', num2str(condition_no), ')')); 
    subplot(1,3,3); imshow(abs(phantom-ima),[]); 
    title('Error Image'); 
    saveas(gcf,strcat('1.6_cond', num2str( cond(k) ), '.png' ));          

    %IQA measurements
    PSNR(k) = psnr(ima,phantom);
    SSIM(k) = ssim(ima,phantom);            
    %disp(strcat('Condition Number: ', num2str(cond(k))));
    disp(strcat('PSNR for Condition Number=', num2str(condition_no), ':',num2str(PSNR(k))));
    disp(strcat('SSIM for Condition Number=', num2str(condition_no), ':',num2str(SSIM(k))));

end

figure;set(gcf, 'WindowState', 'maximized');
subplot(1,2,1); stem(cond,PSNR); xlim([0 60]);
title('PSNR vs. Condition Number');
xlabel('Condition Number'); ylabel('PSNR');
subplot(1,2,2); stem(cond,SSIM); xlim([0 60]);
title('SSIM vs. Condition Number');    
xlabel('Condition Number'); ylabel('SSIM');
saveas(gcf,'1.6_PSNR_SSIM.png');

% best condition number: 20

case '1.7'

disp('1.7');
disp("Filtered SVD");

load("S.mat"); load("u.mat");
load("measurement_phantom.mat",'meas_coil1','meas_coil2','phantom');

[U,Sigma,V] = svd(S,'econ');
singular_values = diag(Sigma);

%lambda: regularization parameter 
%lambda should be much larger than square of minimum singular value
%and much smaller than square of maximum singular value 
%for starting, lambda = sing_max * sing_min;
singular_max = singular_values(1);
singular_min = singular_values(end);    
lambda = singular_max * singular_min;
disp(strcat('Regularization Parameter: ', num2str(lambda)));

%find the new singular values (filtered singular values)
%replace them in new Sigma 
new_Sigma =  Sigma + (lambda./Sigma);

%singular values 
original_singular_values = diag(Sigma);
filtered_singular_values = diag(new_Sigma);

%comparison of original singular values and filtered singular values
%singular values increased for higher frequency components 
%so that the image is effectively low-pass filtered, eliminated noise
figure; set(gcf, 'WindowState', 'maximized');
plot(original_singular_values); hold on; 
plot(filtered_singular_values);
title('Singular Values');
xlabel('i'); ylabel('{\sigma}_i');
legend('Original Singular Values','Filtered Singular Values','magenta','orange','Northwest');
saveas(gcf,'1.7_singular_values.png'); 

c = V*((U'*u)./diag(new_Sigma));

ima = reshape(c,40,40);
ima = imresize(ima, [280 280]);
ima(ima<0) = 0; 

figure;set(gcf, 'WindowState', 'maximized');
subplot(1,3,1); imshow(phantom,[]); 
title('Reference Image');
subplot(1,3,2); imshow(ima,[]); 
title(strcat('Filtered SVD ({\lambda} = ', num2str(lambda), ')')); 
subplot(1,3,3); imshow(abs(phantom-ima),[]); 
title('Error Image'); 
saveas(gcf,strcat('1.7_', num2str(lambda), '.png' ));          

%IQA measurements
PSNR = psnr(ima,phantom);
SSIM = ssim(ima,phantom);            
disp(strcat('PSNR: ', num2str(PSNR)));
disp(strcat('SSIM: ', num2str(SSIM)));

case '1.8'

disp('1.8');
disp("Filtered SVD with Different Lambdas");

load("S.mat"); load("u.mat");
load("measurement_phantom.mat",'meas_coil1','meas_coil2','phantom');

%show steps for finding optimum lambda
for lambda = [100:-10:50].^2
    disp(strcat('Regularization Parameter: ', num2str(lambda)));

    [U,Sigma,V] = svd(S,'econ');
    new_Sigma =  Sigma + (lambda./Sigma);    
    c = V*((U'*u)./diag(new_Sigma));

    ima = reshape(c,40,40);
    ima = imresize(ima, [280 280]);
    ima(ima<0) = 0; 

    figure;set(gcf, 'WindowState', 'maximized');
    subplot(1,3,1); imshow(phantom,[]); 
    title('Reference Image');
    subplot(1,3,2); imshow(ima,[]); 
    title(strcat('Filtered SVD ({\lambda} = ', num2str(lambda), ')')); 
    subplot(1,3,3); imshow(abs(phantom-ima),[]); 
    title('Error Image'); 
    %saveas(gcf,strcat('1.8_step', num2str(lambda), '.png' ));          

    %IQA measurements
    PSNR = psnr(ima,phantom);
    SSIM = ssim(ima,phantom);            
    disp(strcat('PSNR for Regularization Parameter=', num2str(lambda),':',num2str(PSNR)));
    disp(strcat('SSIM for Regularization Parameter=', num2str(lambda),':',num2str(SSIM)));
  
end

%lambda (regularization parameter) 
opt_lambda = 4900;
disp(strcat('Optimum Regularization Parameter: ', num2str(opt_lambda)));

for lambda = [opt_lambda/10 opt_lambda 10*opt_lambda]
    disp(strcat('Regularization Parameter: ', num2str(lambda)));

    [U,Sigma,V] = svd(S,'econ');
    new_Sigma =  Sigma + (lambda./Sigma);    
    c = V*((U'*u)./diag(new_Sigma));

    ima = reshape(c,40,40);
    ima = imresize(ima, [280 280]);
    ima(ima<0) = 0; 

    figure;set(gcf, 'WindowState', 'maximized');
    subplot(1,3,1); imshow(phantom,[]); 
    title('Reference Image');
    subplot(1,3,2); imshow(ima,[]); 
    title(strcat('Filtered SVD ({\lambda} = ', num2str(lambda), ')')); 
    subplot(1,3,3); imshow(abs(phantom-ima),[]); 
    title('Error Image'); 
    saveas(gcf,strcat('1.8_', num2str(lambda), '.png' ));          

    %IQA measurements
    PSNR = psnr(ima,phantom);
    SSIM = ssim(ima,phantom);            
    disp(strcat('PSNR for Regularization Parameter=', num2str(lambda),':',num2str(PSNR)));
    disp(strcat('SSIM for Regularization Parameter=', num2str(lambda),':',num2str(SSIM)));
  
end

case '1.9'

disp('1.9');
disp("L-curve");

load("S.mat"); load("u.mat");
load("measurement_phantom.mat",'meas_coil1','meas_coil2','phantom');

[U,Sigma_ref,V] = svd(S,'econ');

lambda = 100:100:10000;
for i = 1:length(lambda)  
    Sigma =  Sigma_ref + (lambda(i)./Sigma_ref);    
    c = V*((U'*u)./diag(Sigma));              
    residual_norm(i) = norm((S*c - u),2); 
    solution_norm(i) = norm(c,2);
end

figure;set(gcf, 'WindowState', 'maximized');
plot(residual_norm,solution_norm);
title('L-curve');
xlabel('||Sc - u||_2'); ylabel('||c||_2');
saveas(gcf,'1.9_L-curve.png');          

%find the optimum lambda
opt_lambda = lambda(max(find(residual_norm<=453.9)));
[U,Sigma,V] = svd(S,'econ');
new_Sigma =  Sigma + (opt_lambda./Sigma);    
c = V*((U'*u)./diag(new_Sigma));

ima = reshape(c,40,40);
ima = imresize(ima, [280 280]);
ima(ima<0) = 0; 

figure;set(gcf, 'WindowState', 'maximized');
subplot(1,3,1); imshow(phantom,[]); 
title('Reference Image');
subplot(1,3,2); imshow(ima,[]); 
title(strcat('Filtered SVD ({\lambda} = ', num2str(opt_lambda), ')')); 
subplot(1,3,3); imshow(abs(phantom-ima),[]); 
title('Error Image'); 
saveas(gcf,strcat('1.9.png' ));          

%IQA measurements
PSNR = psnr(ima,phantom);
SSIM = ssim(ima,phantom);            
disp(strcat('PSNR for Regularization Parameter=', num2str(opt_lambda),':',num2str(PSNR)));
disp(strcat('SSIM for Regularization Parameter=', num2str(opt_lambda),':',num2str(SSIM)));

%%part2 - kaczmarz method 
case '2.1'

disp('2.1');
disp("Row-norm Thresholding");

load("S.mat"); load("u.mat");

treshold = 50;   
[original_norm_S,norm_S,S,u] = row_norm_treshold(treshold,S,u);  

figure;set(gcf, 'WindowState', 'maximized');
plot(original_norm_S);
title('Row Number vs. Row Norm');
xlabel('Row Number');ylabel('Row Norm'); 
saveas(gcf,'2.1.png');          

size_S = size(S);
size_u = size(u);    
disp(strcat('Size of S: ', num2str(size_S(1)), 'x',num2str(size_S(2)) ));    
disp(strcat('Size of u: ', num2str(size_u(1)), 'x',num2str(size_u(2)) ));

case '2.2'

disp('2.2');
disp("Standard Kaczmarz Method");

load("measurement_phantom.mat",'meas_coil1','meas_coil2','phantom');
load("S.mat"); load("u.mat");

treshold = 50;   
[original_norm_S,norm_S,S,u] = row_norm_treshold(treshold,S,u);  

%randomize rows proportional to ||si||^2 
%take sorting_order as the order of sub-iterations 
[sorted_norm_S,sorting_order] = sort((norm_S.^2),'descend');

[row_no col_no] = size(S);

%start from origin as first estimate solution   
c = zeros(col_no,1);
for iter = 1:10        
    for sub_iter = row_no:-1:1
        ind = find(sorting_order == sub_iter);
        si=S(ind,:);
        ui=u(ind);            
        si_H = si';
        %When A and B are both
        %column vectors, dot(A,B) 
        %is the same as A'*B.
        c = c + ((ui-dot(si,c))/(norm(si,2)^2))*si_H;
    end

    ima = reshape(c,40,40);
    ima = imresize(ima, [280 280]);
    ima(ima<0) = 0; %set negative values to zero         

    figure;set(gcf, 'WindowState', 'maximized');
    subplot(1,3,1); imshow(phantom,[]); 
    title('Reference Image');
    subplot(1,3,2); imshow(ima,[]); 
    title(strcat('Standar Kaczmarz for Iteration #', num2str(iter))); 
    subplot(1,3,3); imshow(abs(phantom-ima),[]); 
    title('Error Image'); 
    saveas(gcf,strcat('2.2_iter', num2str( iter ), '.png' ));          

    %IQA measurements
    PSNR = psnr(ima,phantom);
    SSIM = ssim(ima,phantom);            
    disp(strcat('PSNR for Iteration #',num2str(iter),': ', num2str(PSNR)));
    disp(strcat('SSIM for Iteration #',num2str(iter),': ', num2str(SSIM)));

end  

case '2.3'

disp('2.3');
disp("Standard Kaczmarz Method with Different Tresholds");

load("measurement_phantom.mat",'meas_coil1','meas_coil2','phantom');
load("S.mat"); load("u.mat");
S_ref = S; u_ref = u;

for treshold = [1 10 30 50 100]

    %randomize rows proportional to ||si||^2 
    %take sorting_order as the order of sub-iterations 
    [original_norm_S,norm_S,S,u] = row_norm_treshold(treshold,S_ref,u_ref);  
    [sorted_norm_S,sorting_order] = sort((norm_S.^2),'descend');

    [row_no col_no] = size(S);

    %start from origin as first estimate solution 
    c = zeros(col_no,1);
    for iter = 1:10        
        for sub_iter = row_no:-1:1
            ind = find(sorting_order == sub_iter);
            si=S(ind,:);
            ui=u(ind);            
            si_H = si';
            c = c + ((ui-dot(si,c))/(norm(si,2)^2))*si_H;
        end
    end    

    ima = reshape(c,40,40);
    ima = imresize(ima, [280 280]);
    ima(ima<0) = 0; %set negative values to zero         

    figure;set(gcf, 'WindowState', 'maximized');
    subplot(1,3,1); imshow(phantom,[]); 
    title('Reference Image');
    subplot(1,3,2); imshow(ima,[]); 
    title(strcat('Standar Kaczmarz (Treshold=', num2str(treshold), ')')); 
    subplot(1,3,3); imshow(abs(phantom-ima),[]); 
    title('Error Image'); 
    saveas(gcf,strcat('2.3_treshold', num2str( treshold ), '.png' ));          

    %IQA measurements
    PSNR = psnr(ima,phantom);
    SSIM = ssim(ima,phantom);            
    disp(strcat('PSNR for Treshold= ',num2str(treshold), ': ', num2str(PSNR)));
    disp(strcat('SSIM for Treshold= ',num2str(treshold), ': ', num2str(SSIM)));

end

case '2.4'

disp('2.4');
disp("Regularized Kaczmarz Method");

load("measurement_phantom.mat",'meas_coil1','meas_coil2','phantom');
load("S.mat"); load("u.mat");

treshold = 1;   
[original_norm_S,norm_S,S,u] = row_norm_treshold(treshold,S,u);
%randomize rows proportional to ||si||^2 
[sorted_norm_S,sorting_order] = sort((norm_S.^2),'descend');

%weighting function 
w = 1./(norm_S.^2);
W = diag(w);

[U,Sigma,V] = svd(S,'econ');
singular_values = diag(Sigma);
lambda = singular_values(1)* singular_values(end)

[row_no, col_no] = size(S);
identity = eye(row_no);

%start from origin for first estimate solution 
c = zeros(col_no,1);    
v = zeros(row_no,1);
for iter = 1:10        
    for sub_iter = row_no:-1:1
        ind = find(sorting_order == sub_iter);%find index
        %parameters 
        si=S(ind,:);
        ui=u(ind); 
        si_H = si';
        wi = w(ind);
        ei = identity(:,ind);
        vi = v(ind);
        %algorithm
        a = ( ui - dot(si_H,c) - sqrt(lambda/wi)*vi )./( norm(si,2)^2 + (lambda/wi) ); 
        c = c + a*si_H;
        v = v + a*sqrt(lambda/wi)*ei;
    end

    ima = reshape(c,40,40);
    ima = imresize(ima, [280 280]);
    ima(ima<0) = 0; %set negative values to zero         

    figure;set(gcf, 'WindowState', 'maximized');
    subplot(1,3,1); imshow(phantom,[]); 
    title('Reference Image');
    subplot(1,3,2); imshow(ima,[]); 
    title(strcat('Regularized for Iteration #', num2str(iter))); 
    subplot(1,3,3); imshow(abs(phantom-ima),[]); 
    title('Error Image'); 
    saveas(gcf,strcat('2.4_iter', num2str( iter ), '.png' ));          

    %IQA measurements
    PSNR = psnr(ima,phantom);
    SSIM = ssim(ima,phantom);            
    disp(strcat('PSNR for Iteration #',num2str(iter),': ', num2str(PSNR)));
    disp(strcat('SSIM for Iteration #',num2str(iter),': ', num2str(SSIM)));

end

case '2.5'

disp('2.5');
disp("Regularized Kaczmarz Method with Different Lambdas");

load("measurement_phantom.mat",'meas_coil1','meas_coil2','phantom');
load("S.mat"); load("u.mat");
S_ref = S; u_ref = u;

treshold = 1; 
[original_norm_S,norm_S,S,u] = row_norm_treshold(treshold,S,u);
%randomize rows proportional to ||si||^2 
[sorted_norm_S,sorting_order] = sort((norm_S.^2),'descend');

%weighting function 
w = 1./(norm_S.^2);
W = diag(w);

[U,Sigma,V] = svd(S,'econ');
singular_values = diag(Sigma);
lambda_rel = [1e-4 1e-3 1e-2 1e-1]; 

[row_no, col_no] = size(S);
identity = eye(row_no);


for lambda = (singular_values(1)^2).*lambda_rel

    %randomize rows proportional to ||si||^2 
    [original_norm_S,norm_S,S,u] = row_norm_treshold(treshold,S_ref,u_ref);  
    [sorted_norm_S,sorting_order] = sort((norm_S.^2),'descend');
    [row_no col_no] = size(S);
    %start from origin for first estimate solution 
    c = zeros(col_no,1);    
    v = zeros(row_no,1);
    for iter = 1:10        
        for sub_iter = row_no:-1:1
            ind = find(sorting_order == sub_iter);%find index
            %parameters 
            si=S(ind,:);
            ui=u(ind); 
            si_H = si';
            wi = w(ind);
            ei = identity(:,ind);
            vi = v(ind);
            %algorithm
            a = ( ui - dot(transpose(si_H),c) - sqrt(lambda/wi)*vi )./( norm(si,2)^2 + (lambda/wi) ); 
            c = c + a*si_H;
            v = v + a*sqrt(lambda/wi)*ei;
        end

    end

    ima = reshape(c,40,40);
    ima = imresize(ima, [280 280]);
    ima(ima<0) = 0; %set negative values to zero         

    figure;set(gcf, 'WindowState', 'maximized');
    subplot(1,3,1); imshow(phantom,[]); 
    title('Reference Image');
    subplot(1,3,2); imshow(ima,[]); 
    title(strcat('Regularized Kaczmarz ({\lambda}=',num2str(lambda),')')); 
    subplot(1,3,3); imshow(abs(phantom-ima),[]); 
    title('Error Image'); 
    saveas(gcf,strcat('2.5_lambda', num2str( lambda ), '.png' ));          

    %IQA measurements
    PSNR = psnr(ima,phantom);
    SSIM = ssim(ima,phantom);            
    disp(strcat('PSNR for Lambda=',num2str(lambda),': ', num2str(PSNR)));
    disp(strcat('SSIM for Lambda=',num2str(lambda),': ', num2str(SSIM)));

end 

case '2.6' 

disp('2.6');
disp("Regularized Kaczmarz Method with Different Tresholds for the Best Lambda");

load("measurement_phantom.mat",'meas_coil1','meas_coil2','phantom');

load("S.mat");load("u.mat");
S_ref = S;u_ref = u;

lambda = 1276.1729;

for treshold = [1 10 30 50 100]

    %randomize rows proportional to ||si||^2 
    [original_norm_S,norm_S,S,u] = row_norm_treshold(treshold,S_ref,u_ref);  
    [sorted_norm_S,sorting_order] = sort((norm_S.^2),'descend');

    [U,Sigma,V] = svd(S,'econ');

    %weighting function 
    w = 1./(norm_S.^2);
    W = diag(w);

    [row_no col_no] = size(S);
    identity = eye(row_no);
    %start from origin for first estimate solution 
    c = zeros(col_no,1);    
    v = zeros(row_no,1);
    for iter = 1:10        
        for sub_iter = row_no:-1:1
            ind = find(sorting_order == sub_iter);%find index
            %parameters 
            si=S(ind,:);
            ui=u(ind); 
            si_H = si';
            wi = w(ind);
            ei = identity(:,ind);
            vi = v(ind);
            %algorithm
            a = ( ui - dot(transpose(si_H),c) - sqrt(lambda/wi)*vi )./( norm(si,2)^2 + (lambda/wi) ); 
            c = c + a*si_H;
            v = v + a*sqrt(lambda/wi)*ei;
        end
    end

    ima = reshape(c,40,40);
    ima = imresize(ima, [280 280]);
    ima(ima<0) = 0; %set negative values to zero         

    figure;set(gcf, 'WindowState', 'maximized');
    subplot(1,3,1); imshow(phantom,[]); 
    title('Reference Image');
    subplot(1,3,2); imshow(ima,[]); 
    title(strcat('Regularized Kaczmarz (Treshold=',num2str(treshold),')')); 
    subplot(1,3,3); imshow(abs(phantom-ima),[]); 
    title('Error Image'); 
    saveas(gcf,strcat('2.6_treshold', num2str( treshold ), '.png' ));          

    %IQA measurements
    PSNR = psnr(ima,phantom);
    SSIM = ssim(ima,phantom);            
    disp(strcat('PSNR for Treshold=',num2str(treshold),': ', num2str(PSNR)));
    disp(strcat('SSIM for Treshold=',num2str(treshold),': ', num2str(SSIM)));

end 

case '2.7'
    
disp('2.7');
disp("Effects of Measurement Noise");

%%part 3-open mpi data
case '3.1'

disp('3.1');
load("su_openmpi.mat",'S','u');

[U,Sigma,V] = svd(S,'econ');
%plot the singular values 
singular_values = diag(Sigma);
figure;set(gcf, 'WindowState', 'maximized');
plot(singular_values); title('Singular Values');
xlabel('i'); ylabel('{\sigma}_i');
saveas(gcf,'3.1_singular_values.png');   

condition_no = singular_values(1)/singular_values(end);
disp(strcat('Condition Number: ', num2str(condition_no))); 

case '3.2'
disp('3.2');
load("su_openmpi.mat",'S','u');

treshold = 30;   
[original_norm_S,norm_S,S,u] = row_norm_treshold(treshold,S,u);
%randomize rows proportional to ||si||^2 
[sorted_norm_S,sorting_order] = sort((norm_S.^2),'descend');

%weighting function 
w = 1./(norm_S.^2);
W = diag(w);

[U,Sigma,V] = svd(S,'econ');
singular_values = diag(Sigma);
lambda = singular_values(1)* singular_values(end);

[row_no, col_no] = size(S);
identity = eye(row_no);

%start from origin for first estimate solution 
c = zeros(col_no,1);    
v = zeros(row_no,1);
for iter = 1:10        
    for sub_iter = row_no:-1:1
        ind = find(sorting_order == sub_iter);%find index
        %parameters 
        si=S(ind,:);
        ui=u(ind); 
        si_H = si';
        wi = w(ind);
        ei = identity(:,ind);
        vi = v(ind);
        %algorithm
        a = ( ui - dot(si_H,c) - sqrt(lambda/wi)*vi )./( norm(si,2)^2 + (lambda/wi) ); 
        c = c + a*si_H;
        v = v + a*sqrt(lambda/wi)*ei;
    end
end

ima = reshape(c,37,37,37);
ima(ima<0) = 0; %set negative values to zero     
figure;set(gcf, 'WindowState', 'maximized');
montage(reshape(ima,[37, 37, 1, 37]),'displayRange',[]);
saveas(gcf,'3.2.png' );          

case '3.3'
    
disp('3.3');
load("su_openmpi.mat",'S','u');

treshold = 30;   
[original_norm_S,norm_S,S,u] = row_norm_treshold(treshold,S,u);
%randomize rows proportional to ||si||^2 
[sorted_norm_S,sorting_order] = sort((norm_S.^2),'descend');

%weighting function 
w = 1./(norm_S.^2);
W = diag(w);

[U,Sigma,V] = svd(S,'econ');
singular_values = diag(Sigma);
lambda = 1;

[row_no, col_no] = size(S);
identity = eye(row_no);

%start from origin for first estimate solution 
c = zeros(col_no,1);    
v = zeros(row_no,1);
for iter = 1:10        
    for sub_iter = row_no:-1:1
        ind = find(sorting_order == sub_iter);%find index
        %parameters 
        si=S(ind,:);
        ui=u(ind); 
        si_H = si';
        wi = w(ind);
        ei = identity(:,ind);
        vi = v(ind);
        %algorithm
        a = ( ui - dot(si_H,c) - sqrt(lambda/wi)*vi )./( norm(si,2)^2 + (lambda/wi) ); 
        c = c + a*si_H;
        v = v + a*sqrt(lambda/wi)*ei;
    end
end

ima = reshape(c,37,37,37);
ima(ima<0) = 0; %set negative values to zero     
figure;set(gcf, 'WindowState', 'maximized');
montage(reshape(ima,[37, 37, 1, 37]),'displayRange',[]);
saveas(gcf,'3.3.png' );  
    
case '3.4'
    
disp('3.4');
load("su_openmpi.mat",'S','u');

treshold = 30;   
[original_norm_S,norm_S,S,u] = row_norm_treshold(treshold,S,u);
%randomize rows proportional to ||si||^2 
[sorted_norm_S,sorting_order] = sort((norm_S.^2),'descend');

%weighting function 
w = 1./(norm_S.^2);
W = diag(w);

[U,Sigma,V] = svd(S,'econ');
singular_values = diag(Sigma);
lambda = 1e10;

[row_no, col_no] = size(S);
identity = eye(row_no);

%start from origin for first estimate solution 
c = zeros(col_no,1);    
v = zeros(row_no,1);
for iter = 1:10        
    for sub_iter = row_no:-1:1
        ind = find(sorting_order == sub_iter);%find index
        %parameters 
        si=S(ind,:);
        ui=u(ind); 
        si_H = si';
        wi = w(ind);
        ei = identity(:,ind);
        vi = v(ind);
        %algorithm
        a = ( ui - dot(si_H,c) - sqrt(lambda/wi)*vi )./( norm(si,2)^2 + (lambda/wi) ); 
        c = c + a*si_H;
        v = v + a*sqrt(lambda/wi)*ei;
    end
end

ima = reshape(c,37,37,37);
ima(ima<0) = 0; %set negative values to zero     
figure;set(gcf, 'WindowState', 'maximized');
montage(reshape(ima,[37, 37, 1, 37]),'displayRange',[]);
saveas(gcf,'3.4.png' );  

end

end

function d = fftc(im)
% d = fftc(im)
%
% fftc performs a centered fft2
d = fftshift(fft(ifftshift(im)));
end

function [original_norm_S,norm_S,S,u] = row_norm_treshold(treshold,S,u)   

[row_no col_no] = size(S);   
original_norm_S = zeros(1,row_no);
for i = 1:row_no
    original_norm_S(i) = norm(S(i,:),2);
end    

index = find(original_norm_S<treshold);
S(index,:) = [];
u(index,:) = [];

[row_no col_no] = size(S); 
norm_S = zeros(1,row_no);
for i = 1:row_no
    norm_S(i) = norm(S(i,:),2);
end  

end