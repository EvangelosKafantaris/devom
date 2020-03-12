function [Out_DispEn, npdf]=LinInterDisEn(x,m,nc,MA,tau)
%% This is an implementation of the Linearly Interpolated Dispersion Entropy (LinInterDisEn)
% which is a variation of the original Dispersion Entropy algorithm.

% The LinInterDisEn variation uses linear interpolation to replace missing (NaN)
% samples based on the nearby available samples.

% This variation was produced by 
% Evangelos Kafantaris and Javier Escudero
% evangelos.kafantarise@ed.ac.uk, javier.escudero@ed.ac.uk
% 09/08/2019

% If you use this code, please make sure that you cite the references [1], [2] and [3].
% [1] E. Kafantaris, I. Piper, T.-Y. M. Lo, and J. Escudero, “Augmentation of Dispersion Entropy for Handling Missing and Outlier Samples in Physiological Signal Monitoring,” Entropy, vol. 22, no. 3, p. 319, Mar. 2020, doi: 10.3390/e22030319.
% [2] H. Azami and J. Escudero, "Amplitude- and Fluctuation-based Dispersion Entropy", Entropy, 2018.
% [3] M. Rostaghi and H. Azami, "Dispersion Entropy: A Measure for Time-Series Analysis", IEEE Signal Processing Letters. vol. 23, n. 5, pp. 610-614, 2016.

%% Description of the original Dispersion Entropy algorithm as written by
%  Hamed Azami and Javier Escudero follows:

% This function calculates dispersion entropy (DispEn) of a univariate
% signal x, using different mapping approaches (MA)
%
% Inputs:
%
% x: univariate signal - a vector of size 1 x N (the number of sample points)
% m: embedding dimension
% nc: number of classes (it is usually equal to a number between 3 and 9 - we used c=6 in our studies). Worth noting that the maximum number of classes for the code is 9.
% MA: mapping approach, chosen from the following options (we used 'LOGSIG' in [1]):
% 'LM' (linear mapping),
% 'NCDF' (normal cumulative distribution function),
% 'TANSIG' (tangent sigmoid),
% 'LOGSIG' 'logarithm sigmoid',
% 'SORT' (sorting method).
%
% tau: time lag (it is usually equal to 1)
%
% Outputs:
%
% Out_DispEn: scalar quantity - the DispEn of x
% npdf: a vector of length nc^m, showing the normalized number of dispersion patterns of x
%
% Example: DispEn(rand(1,200),2,6,'LOGSIG',1)
%
% Hamed Azami and Javier Escudero Rodriguez
% hamed.azami@ed.ac.uk and javier.escudero@ed.ac.uk

% The original algorithm was published in 15-January-2018

%%
% The following line is unique to the LinInterDisEn variation and 
% adds the respective linear interpolation mechanism for missing (NaN)
% samples
x = fillmissing(x,'linear');

% The rest of the following code remains the same as in the original
% implementation of Dispersion Entropy
N=length(x);
sigma_x = std(x);
mu_x = mean(x);

%% Mapping approaches

switch MA
    case   'LM'
        y=mapminmax(x,0,1);
        y(y==1)=1-1e-10;
        y(y==0)=1e-10;
        z=round(y*nc+0.5);
        
    case 'NCDF'
        y=normcdf(x,mu_x,sigma_x);
        y=mapminmax(y,0,1);
        y(y==1)=1-1e-10;
        y(y==0)=1e-10;
        z=round(y*nc+0.5);
        
    case 'LOGSIG'
        y=logsig((x-mu_x)/sigma_x);
        y=mapminmax(y,0,1);
        y(y==1)=1-1e-10;
        y(y==0)=1e-10;
        z=round(y*nc+0.5);
        
    case 'TANSIG'
        y=tansig((x-mu_x)/sigma_x)+1;
        y=mapminmax(y,0,1);
        y(y==1)=1-1e-10;
        y(y==0)=1e-10;
        z=round(y*nc+0.5);
        
    case 'SORT'
        x=x(1:nc*floor(N/nc));
        N=length(x);
        [sx osx]=sort(x);
        Fl_NC=N/nc;
        cx=[];
        for i=1:nc
            cx=[cx i*ones(1,Fl_NC)];
        end
        for i=1:N
            z(i)=cx(osx==i);
        end
        
end
%%


all_patterns=[1:nc]';

for f=2:m
    temp=all_patterns;
    all_patterns=[];
    j=1;
    for w=1:nc
        [a,b]=size(temp);
        all_patterns(j:j+a-1,:)=[temp,w*ones(a,1)];
        j=j+a;
    end
end

for i=1:nc^m
    key(i)=0;
    for ii=1:m
        key(i)=key(i)*10+all_patterns(i,ii);
    end
end


embd2=zeros(N-(m-1)*tau,1);
for i = 1:m
    embd2=[z(1+(i-1)*tau:N-(m-i)*tau)]'*10^(m-i)+embd2;
end

pdf=zeros(1,nc^m);

for id=1:nc^m
    [R,C]=find(embd2==key(id));
    pdf(id)=length(R);
end

npdf=pdf/(N-(m-1)*tau);
p=npdf(npdf~=0);
Out_DispEn = -sum(p .* log(p));