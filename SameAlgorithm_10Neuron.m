clear                                           
clc    

load('AppendixC_input_x.mat')  %load data set from file in directory
load('AppendixC_output_d.mat')
 
%Whidden3=[Th3;w13;w23;w33];  
Whidden3=rands(4);               
%Whidden4=[Th4;w14;w24;w34];
Whidden4=rands(4);
Whidden5=rands(4);
Whidden6=rands(4);
Whidden7=rands(4);
Whidden8=rands(4);
Whidden9=rands(4);
Whidden10=rands(4);
Whidden11=rands(4);
Whidden12=rands(4);
Whidden13=rands(4);


W13_op=rands(11);              % 10 neurons in the hidden layer

alpha=0.1;
precision=10^-6;
samples=200;
epoch=0;
epoch_limit=3000;
ebison=2;
mse_Accumalator=0;
MSE=zeros(1000,1);

while (epoch<epoch_limit && ebison>=precision)     
    epoch=epoch+1;
  for i=1:samples
    %% %Forward propagation
    y3=sig((Whidden3'*[-1;x{i}]));  %Y3=g(sum->wji.xi) EQU(5.4)BOOK
    y4=sig((Whidden4'*[-1;x{i}]));  %Y4=g(sum->wji.xi) EQU(5.5)BOOK
    y5=sig((Whidden3'*[-1;x{i}]));  
    y6=sig((Whidden4'*[-1;x{i}])); 
    y7=sig((Whidden4'*[-1;x{i}]));  
    y8=sig((Whidden3'*[-1;x{i}])); 
    y9=sig((Whidden4'*[-1;x{i}]));  
    y10=sig((Whidden4'*[-1;x{i}])); 
    y11=sig((Whidden3'*[-1;x{i}]));  
    y12=sig((Whidden4'*[-1;x{i}]));  
    y13=sig((W13_op'*[-1;y3;y4;y5;y6;y7;y8;y9;y10;y11;y12]));    %Y14=g(sum->wji.yi) EQU(5.6)BOOK

    
    e=d{i}-y13;  %(dj - Yj)
    mse_Accumalator=mse_Accumalator+e^2;
    
    %% %Backword propagation
    
    % partialE/partialW =-(dj - Yj).g'(ij).Yi EQU(5:13)BOOK ,get gradient and adjust in the opposite
   % direction yi here is output and also input to the next neuron 
    
    gradient_13=e*der_sig(y13);   % delta5 = (dj - Y5).g'(i5)  EQU(5.15)BOOK
    gradient_3=W13_op(2)*gradient_13*der_sig(y3); %delta3 = w35.g'(i3) EQU(5.26)
    gradient_4=W13_op(3)*gradient_13*der_sig(y4); %delta4 = w45.g'(i4) EQU(5.26)
    gradient_5=W13_op(4)*gradient_13*der_sig(y5); 
    gradient_6=W13_op(5)*gradient_13*der_sig(y6); 
    gradient_7=W13_op(6)*gradient_13*der_sig(y7); 
    gradient_8=W13_op(7)*gradient_13*der_sig(y8); 
    gradient_9=W13_op(8)*gradient_13*der_sig(y9); 
    gradient_10=W13_op(9)*gradient_13*der_sig(y10); 
    gradient_11=W13_op(10)*gradient_13*der_sig(y11); 
    gradient_12=W13_op(11)*gradient_13*der_sig(y12); 
 
    
     deltaW_th3=alpha*[-1;x{i}]*gradient_3 ; 
     deltaW_th4=alpha*[-1;x{i}]*gradient_4 ;
     deltaW_th5=alpha*[-1;x{i}]*gradient_5 ; 
     deltaW_th6=alpha*[-1;x{i}]*gradient_6 ; 
     deltaW_th7=alpha*[-1;x{i}]*gradient_7 ; 
     deltaW_th8=alpha*[-1;x{i}]*gradient_8 ; 
     deltaW_th9=alpha*[-1;x{i}]*gradient_9 ; 
     deltaW_th10=alpha*[-1;x{i}]*gradient_10 ; 
     deltaW_th11=alpha*[-1;x{i}]*gradient_11 ; 
     deltaW_th12=alpha*[-1;x{i}]*gradient_12 ; 
     deltaW_th13=alpha*[-1;x{i}]*gradient_13 ; 
   
    deltaW_th5_w13s=[ alpha*-1*gradient_13 ; alpha*y3*gradient_13 ; alpha*y4*gradient_13 ; alpha*y5*gradient_13 ; alpha*y6*gradient_13; alpha*y7*gradient_13 ; alpha*y8*gradient_13; alpha*y9*gradient_13 ; alpha*y10*gradient_13; alpha*y11*gradient_13 ; alpha*y12*gradient_13];  %D_W_output5=Alpha.delta5.yi EQU(5.14)

    %%New values of weights
    Whidden3=Whidden3+deltaW_th3; %W_hidden3_new= %W_hidden3_old + Delta_W_hidden3 EQU(5.28)
    Whidden4=Whidden4+deltaW_th4; %W_hidden4_new= %W_hidden4_old + Delta_W_hidden4 EQU(5.28)
    Whidden5=Whidden5+deltaW_th5; )
    Whidden6=Whidden6+deltaW_th6;  
    Whidden7=Whidden7+deltaW_th7; 
    Whidden8=Whidden8+deltaW_th8; 
    Whidden9=Whidden9+deltaW_th9; 
    Whidden10=Whidden10+deltaW_th10; 
    Whidden11=Whidden11+deltaW_th11; 
    Whidden12=Whidden12+deltaW_th12; 
    
    W13_op=W13_op+deltaW_th5_w13s;  %W_out5_new= %W_out5_old + Delta_W_out5 EQU(5.17)
  end
  %%  BEGIN MSE algorithm 
  
 MSE(epoch)=mse_Accumalator/samples; % save valuse of each MSE of each epoch , E_bar=E/P EQU(5.8)BOOK ,Values of table 5.2 stored here in MSE variable
    if(epoch==1)
        ebison=abs(mse_Accumalator/samples) ;  % Get precision value 
        swap=mse_Accumalator/samples;     
    else
        ebison=abs(mse_Accumalator/samples-swap); % Get precision value=abs(Error(current)_Error(previous))
        swap=mse_Accumalator/samples;
    end   
      mse_Accumalator=0;      %reset accumlator 
end 

%% %Plot the graph of E vs Epoch

plot(MSE); 
xlim('auto');
ylim('auto');
xlabel('Epoches');
ylabel('MSE');
grid on;

%% %Notify the user if network converges or not
if(epoch~=epoch_limit)
       disp('precision goal met at epoch'); 
       disp(epoch);
   else
       disp('max epoch reached ');
end
%% Use final weights to test the network with data that the network havent seen before 

TestSetOut_contener=zeros(20,1); %for memory allocation 
e_test=zeros(20,1);                                                 %   test data set 
test={[ 0.0611; 0.2860;0.746],[0.5102;0.7464 ;0.0860 ], [0.0004;0.6916 ;0.5006 ],[0.9430 ;0.4476 ;0.2648],[ 0.1399;0.1610 ;0.2477 ],[0.6423 ;0.3229 ;0.8567 ],[0.6492 ;0.0007 ;0.6422],[0.1818 ;0.5078 ;0.9046 ],[0.7382 ;0.2647 ;0.1916 ], [0.3879 ;0.1307 ;0.8656 ], [0.1903 ;0.6523 ;0.7820 ], [0.8401 ;0.4490 ;0.2719 ],[0.0029 ;0.3264 ;0.2476 ],[0.7088 ;0.9342 ;0.2763 ], [0.1283 ;0.1882 ;0.7253], [0.8882 ;0.3077 ;0.8931 ], [0.2225 ;0.9182 ;0.7820 ], [0.1957 ;0.8423 ;0.3085 ], [0.9991 ;0.5914 ;0.3933 ], [0.2299 ;0.1524 ;0.7353 ]};
test_d={0.4831,0.5965,0.5318,0.6843,0.2872,0.7663,0.5666,0.6601,0.5427,0.5836,0.6950,0.6790,0.2956,0.7742,0.4662,0.8093,0.7581,0.5826,0.7938,0.5012};

    for i=1:20
        t3=sig((Whidden3'*[-1;test{i}]));
        t4=sig((Whidden4'*[-1;test{i}]));
        t5=sig((Whidden5'*[-1;test{i}]));
        t6=sig((Whidden6'*[-1;test{i}]));
        t7=sig((Whidden7'*[-1;test{i}]));
        t8=sig((Whidden8'*[-1;test{i}]));
        t9=sig((Whidden9'*[-1;test{i}]));
        t10=sig((Whidden10'*[-1;test{i}]));
        t11=sig((Whidden11'*[-1;test{i}]));
        t12=sig((Whidden12'*[-1;test{i}]));
        t13=sig((W13_op'*[-1;t3;t4;t5;t6;t7;t8;t9;t10;t11;t12]));
        TestSetOut_contener(i)=sig((W13_op'*[-1;t3;t4;t5;t6;t7;t8;t9;t10;t11;t12])); %Values of table 5.3 stored here in TestSetOut-contener variable
   
     %% new value of error 
     e_test(i)=test_d{i}-t5;
   
          
    end