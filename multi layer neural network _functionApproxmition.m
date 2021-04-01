clear                                           
clc
   
load('AppendixC_input_x.mat')  %load data set from file in directory
load('AppendixC_output_d.mat')
 
%Whidden3=[0.6400;0.4367;0.9373;0.0627];  % ---> %uncomment and replace with initial values in the pdf to test table values
Whidden3=rands(4);               
%Whidden4=[0.3497;0.7887;0.2219;0.5576];
Whidden4=rands(4);
%W5_op=[-0.1531;-0.8184;-0.4671]; 
W5_op=rands(3);              % 2 neurons in the hidden layer

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
    y5=sig((W5_op'*[-1;y3;y4]));    %Y5=g(sum->wji.yi) EQU(5.6)BOOK
    
    e=d{i}-y5;  %(dj - Yj)
    mse_Accumalator=mse_Accumalator+e^2;
    
    %% %Backword propagation
    
    % partialE/partialW =-(dj - Yj).g'(ij).Yi EQU(5:13)BOOK ,get gradient and adjust in the opposite
   % direction yi here is output and also input to the next neuron 
    
    Gradient_5=e*der_sig(y5);   % delta5 = (dj - Y5).g'(i5)  EQU(5.15)BOOK
    Gradient_3=W5_op(2)*Gradient_5*der_sig(y3); %delta3 = w35.g'(i3) EQU(5.26)
    Gradient_4=W5_op(3)*Gradient_5*der_sig(y4); %delta4 = w45.g'(i4) EQU(5.26)
    
    deltaW_th3_w13_w23_w33=alpha*[-1;x{i}]*Gradient_3 ; %D_W_hidden3 = alpha.delta3.xi  EQU(5.25)
    deltaW_th4_w14_w24_w34=alpha*[-1;x{i}]*Gradient_4 ; %D_W_hidden4 = alpha.delta4.xi  EQU(5.25)
    deltaW_th5_w35_w45=[ alpha*-1*Gradient_5 ; alpha*y3*Gradient_5 ; alpha*y4*Gradient_5 ];  %D_W_output5=Alpha.delta5.yi EQU(5.14)

    %%New values of weights
    Whidden3=Whidden3+deltaW_th3_w13_w23_w33; %W_hidden3_new= %W_hidden3_old + Delta_W_hidden3 EQU(5.28)
    Whidden4=Whidden4+deltaW_th4_w14_w24_w34; %W_hidden4_new= %W_hidden4_old + Delta_W_hidden4 EQU(5.28)
    W5_op=W5_op+deltaW_th5_w35_w45;  %W_out5_new= %W_out5_old + Delta_W_out5 EQU(5.17)
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
        t5=sig((W5_op'*[-1;t3;t4]));
        TestSetOut_contener(i)=sig((W5_op'*[-1;t3;t4])); %Values of table 5.3 stored here in TestSetOut-contener variable
   
     %% new value of error 
     e_test(i)=test_d{i}-t5;
   
          
    end


