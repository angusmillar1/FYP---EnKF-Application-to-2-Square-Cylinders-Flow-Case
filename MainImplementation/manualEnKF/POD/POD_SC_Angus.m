clear all
close all
clc


Y = readmatrix('velocity_data.csv', 'Delimiter', ';'); % Y is the snapshot matrix, 
N = size(Y,1); % N = 36966 / mesh2 / 168506 
W = load("cellVolumes_mesh1_doubled.txt"); % Weighting matrix
% W = diag(Wvect);


dt = 1.0;
T = 0:dt:(size(Y,2)-1)*dt;
K = length(T);

Y2 = sqrt(W).*Y; % Multiplying the snapshot matrix by the weighting matrix V^(1/2)*Y 

[PHI,SIGMA] = svd(Y2,'econ'); % Singular value decomposition
    
% Eigenvalues (energy of POD modes)
    lambda = diag(SIGMA).^2/K;
    
% Eigenvectors (spatial POD modes)
    % phi = (1/sqrt(dx*dy))*PHI;
    W2 = 1./sqrt(W);
    phi = W2.*PHI;  % Multiplying by to eliminate weighting and display phi on a grid V^(-1/2)*Y    


    % POD temporal coefficients
    %a = (Y'*sqrt(dx*dy)*PHI)';
    a = (transpose(Y2)*PHI)';
    
    % Orthogonality of POD coefficients
    orthogonality_check = zeros(10,10); % pre-allocation
    for i = 1:10
        for j = 1:10
            orthogonality_check(i,j) = round(dot(a(i,:),a(j,:)),2);
        end
    end


%% Energy of POD Modes

eigenvals_POD = figure('Name','Energy spectrum of POD modes'); 

% First subplot
subplot(1,2,1);
plot(1:length(lambda), lambda / sum(lambda), '-o', 'Linewidth', 1, ...
    'Color', 'k', 'Markersize', 4, 'MarkerFaceColor', 'k', ...
    'MarkerEdgeColor', [0.5 0.5 0.5]);
hold on;
% Overlay the first 5 indices in red
plot(1:5, lambda(1:5) / sum(lambda), '-o', 'Linewidth', 1, 'Color', 'r', ...
    'Markersize', 4, 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'r');
xlabel('Mode $i$', 'fontsize', 18, 'Interpreter', 'Latex'); 
ylabel('$\lambda_{i}/\sum_{i=1}^{K}\lambda_i$', 'fontsize', 18, 'Interpreter', 'Latex'); 
grid on;
xlim([1 150]); 
ylim([0 0.05]);
set(gca, 'FontSize', 18, 'FontName', 'Courier');
legend('$i \in [1, 5]$', '$i \in [6, 150]$', 'Interpreter', 'Latex', 'FontSize', 18, 'Location', 'best');
hold off;

% Second subplot
subplot(1,2,2);
plot(1:5, lambda(1:5) / sum(lambda), 'r-o', 'Linewidth', 1, 'Markersize', 4, ...
    'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'r'); 
hold on;
plot(6:length(lambda), lambda(6:end) / sum(lambda), '-o', 'Linewidth', 1, ...
    'Color', [0.5 0.5 0.5], 'Markersize', 4, 'MarkerFaceColor', [0.5 0.5 0.5], ...
    'MarkerEdgeColor', [0.5 0.5 0.5]);
xlabel('Mode $i$', 'fontsize', 18, 'Interpreter', 'Latex'); 
ylabel('$\lambda_{i}/\sum_{i=1}^{K}\lambda_i$', 'fontsize', 18, 'Interpreter', 'Latex'); 
grid on; 
xlim([1 5]);
legend('$i \in [1, 5]$', '$i \in [6, 150]$', 'Interpreter', 'Latex', 'FontSize', 18, 'Location', 'best');
set(gca, 'FontSize', 18, 'FontName', 'Courier');
x0 = 10;
y0 = 10;
width = 1200;
height = 600;
set(gcf, 'position', [x0, y0, width, height]);



% plot(1:50,100*cumsum(lambda(1:50)/sum(lambda)),'r-o','Linewidth',1,'Markersize',4,'MarkerFaceColor','r','MarkerEdgeColor','r'); hold on;
% energy_truncated = cumsum(lambda(1:50)/sum(lambda));
% plot(51:length(lambda),100*(energy_truncated(end)+cumsum(lambda(51:end)/sum(lambda))),'-o','Linewidth',1,'Color',[0.5 0.5 0.5],'Markersize',4,'MarkerFaceColor',[0.5 0.5 0.5],'MarkerEdgeColor',[0.5 0.5 0.5]);
% xlabel('Mode $i$','fontsize',15,'Interpreter','Latex'); 
% ylabel('Cumulative Energy fraction (\%)','fontsize',15,'Interpreter','Latex'); 
% grid on; xlim([1 150]); xticks([0:25:150]); ylim([0 100]);
% yBox = [0, 0, 100, 100, 0];
% xBox = [1, 50, 50, 1, 1];
% patch(xBox, yBox, 'red', 'FaceColor', 'red', 'EdgeColor','none','FaceAlpha', 0.2); 
% xBox = [50, 150, 150, 50, 50];
% patch(xBox, yBox, 'black', 'FaceColor', [0.5 0.5 0.5], 'EdgeColor','none','FaceAlpha', 0.2);

% grid on
% set(gca,'FontSize',14, 'FontName', 'Courier')

% eigenvals_POD.Position = [100 100 1200 500];
%%
% writematrix(phi(:,1),'../phi_1.txt');
% writematrix(phi(:,2),'../phi_2.txt');
% writematrix(phi(:,3),'../phi_3.txt');
% writematrix(phi(:,4),'../phi_4.txt');
% writematrix(phi(:,5),'../phi_5.txt');
% writematrix(phi(:,6),'../phi_6.txt');
% writematrix(phi(:,7),'../phi_7.txt');
% writematrix(phi(:,8),'../phi_8.txt');
% writematrix(phi(:,9),'../phi_9.txt');
% writematrix(phi(:,10),'../phi_10.txt');

%% POD temporal coefficients
title_fig = ["POD temporal coefficients 1-5","POD temporal coefficients 6-10","POD temporal coefficients 11-15",...
    "POD temporal coefficients 16-20","POD temporal coefficients 21-25","POD temporal coefficients 26-30",...
    "POD temporal coefficients 31-35","POD temporal coefficients 36-40","POD temporal coefficients 41-45",...
    "POD temporal coefficients 46-50"];
for j = 1
    switch j
        case 1
            i = 1;
        case 2
            i = 5:8;
        case 3
            i = 11:15;
        case 4
            i = 16:20;
        case 5
            i = 21:25;
        case 6
            i = 26:30;
        case 7
            i = 31:35;
        case 8
            i = 36:40;
        case 9
            i = 41:45;
        case 10
            i = 46:50;
    end
    figure('Name',title_fig(j));
    
    for n = 1:length(i)

        % % Time domain
        % subplot(1,2,2*n);
        % plot(T,a(i(n),:),'b-','Linewidth',1);
        % grid on; xlim([T(1) T(end)]);
        % ylabel(['$a_{',num2str(i(n)),'}(t)$'],'fontsize',14,'interpreter','latex');
        % if n == 5
        %     xlabel('Time','fontsize',14,'interpreter','latex');
        % end
        % set(gca,'Fontsize',14);
        
        % Frequency domain
        figure(2)

        Windows = 11; 
        LW = 1;
        SegmentLength = size(a(i(n),:)',1)/Windows;
        Fs = 2;
        Hs = spectrum.welch('Hamming',SegmentLength,20);
        W = psd(Hs,a(i(n),:),'Fs',Fs);
        P1 = W.Data;
        f = W.Frequencies;

        plot(f,P1,'LineWidth',1.5,'DisplayName',"$a_{"+num2str(i(n))+"}$");
        % xlim([0 0.25]);
        % ylim([0 0.005]);
        grid on
        hold on;
        ylabel(['Power Spectral Density'],'fontsize',18,'interpreter','latex');
        xlabel('$St$','fontsize',18,'interpreter','latex');
        legend('show','FontSize',18,'interpreter','latex') %,'Location','southoutside')
        x0=10;
        y0=10;
        width=600;
        height=700;
        set(gcf,'position',[x0,y0,width,height])
        set(gca,'FontSize',18,'FontName', 'Courier')

        
        % [PSD] = spectrum_analyser([T',a(i(n),:)'],0);
        % s = plot(PSD(1,:),PSD(2,:),'b-','Linewidth',1); hold on;
        % grid on; 
        % ylim([0 0.01]);
        % title("Frequency spectrum of $C_L$",'Interpreter','Latex','Fontsize',15);
        % xlabel('$St$','Interpreter','Latex','Fontsize',15);
        % ylabel("FFT amplitude of $C_L$",'Interpreter','Latex','Fontsize',15)
        % insert datatip
        % [~,indx] = max(PSD(2,:));
        % dt = datatip(s,'DataIndex',indx,'interpreter','latex','fontsize',16);
        % s.DataTipTemplate.DataTipRows(1).Label = '$St$'; s.DataTipTemplate.DataTipRows(1).Format = '%0.2f';
        % s.DataTipTemplate.DataTipRows(2).Label = '$|a(St)|$'; s.DataTipTemplate.DataTipRows(2).Format = '%0.3f';
        % if n == 5
        %     xlabel('$St$','fontsize',18,'interpreter','latex');
        % end
        
    end
end