close all
clear all
                %%%%  NOTES  %%%%
% >>> all units used here are in meters due to the ADK rate being
% calculated with E(GV/m)
% >>> The wavelength is hard coded in the function M2_fit
% >>> The Z axis origin is the location where the delay stage is the
% closest to the motor. 
% 

%  PLEASE SEE THE USAGE MANUAL FOR FURTHER EXPLANATIONS, DATA FLOW, PICO
%  MOTOR COMMANDS ETC


%%%%  USER INPUT %%%%

%%%%  PLOTTING/TESTING/SAVING OPTIONS %%%%

% Defining directory to save plasma density distorbution
% figure_path ='nfs://flashhome1/user/fflab28m/Matlab/ldickson/plots';
% raw_plasma_density_ratio_path = 'nfs://flashhome1/user/fflab28m/Matlab/ldickson/raw_plasma_data';

%  PATHS FOR LOCAL DRIVE OF LEWIS DICKSON - ldickson
figure_path = '/home/ldickson/documents/MATLAB/data/plots';
raw_plasma_density_ratio_path = '/home/ldickson/documents/MATLAB/data/raw_plasma_data';
raw_image_fig_path = '/home/ldickson/documents/MATLAB/data/raw_image_fig';
raw_image_data_path = '/home/ldickson/documents/MATLAB/data/raw_image_data';
image_projection_path = '/home/ldickson/documents/MATLAB/data/image_projection';
sigma_plot_path = '/home/ldickson/documents/MATLAB/data/sigma_plots';
background_image_path = '/home/ldickson/Documents/final/ldickson/PlasmaDensity/background_image.mat';
Msquared_path = '/home/ldickson/documents/MATLAB/data/Msquared_plots';
real_time_energy_path = '/home/ldickson/documents/MATLAB/data/real_time_energy';
full_plasma_ratio_data = '/home/ldickson/documents/MATLAB/data/full_plasma_ratio_data';
% TESTING PATHS ON DESKTOP
% figure_path = '/Users/fflab28m/Desktop/temp_folder_ldickson/Updated_script/plots';
% raw_plasma_density_ratio_path = '/Users/fflab28m/Desktop/temp_folder_ldickson/Updated_script/raw_plasma_data';
% raw_image_fig_path = '/Users/fflab28m/Desktop/temp_folder_ldickson/Updated_script/raw_image_fig';
% raw_image_data_path = '/Users/fflab28m/Desktop/temp_folder_ldickson/Updated_script/raw_image_data';
% image_projection_path = '/Users/fflab28m/Desktop/temp_folder_ldickson/Updated_script/image_projection';
% sigma_plot_path = '/Users/fflab28m/Desktop/temp_folder_ldickson/Updated_script/sigma_plots';

% background_image_path = '/home/ldickson/documents/final/ldickson_background_images/';
% background_image_type = '*.pgm';  %use wild card so that dir doesn't pick up '.' or '..' in directory

%%%%  CODE OPTIONS  %%%%

% Using gaussian distorbution of intensities for testing
example_setting = 2; % 0 for real data, 1 for guassian example, 2 for loading already taken data (put taken data in working directory) 

% Defining plotting options
% to reopen use openfig('PATH TO FILE NAME INSERT HERE','new','visible')
hide_fig = 1; % a value of '1' will not display any figures but it will still be saved

% Choose plasma density ratio or absolute plasma density
plasma_choice = 1; % 1 gives the ratio of plasma to neutral gas, otherwise absolute plasama density is calculated

%%%%  PLASMA/LASER OPTIONS %%%%

% Define the plasma characteristics
ionisation_number = 1; % Charge of ion after ionisation -> 1 for single removed electron from neutral atom
% Ki_i =13.6;%energy of unperturbed electron ground state in eV for hydrogen
Ki_i = 15.75962./27.2; %energy of unperturbed electron ground state in eV for argon normalised to atomic energy scale

if plasma_choice == 1
    gas_density = 1;  % PLEASE DO NOT CHANGE
else
    gas_density = 10^14; % INPUT GAS DENSITY HERE - Gas density required for absolute plasma density calculation
end

% Define laser characteristics
tau = 30*10^-15; %pulse length
lamda = 800*10^-9;

%%%%  DATA ANALYSIS OPTIONS  %%%%

% Define the number of data slices and range of values - Scan stage values
% - works for limits between 323000 - 4120000
max_val = 0; % in motor input steps
min_val = -3810000; 
num_data_slices = 100;
step_size = 1.05*10^-6;  % steps 1.05 um

only_take_data = 0; % value of 1 will only take and save intensity data with no data analysis 
remove_background = 1; %value of 1 will remove the average background from the read image.
calculate_plasma_density_ratio = 1; % value of 1 will calculate the plasma density ratio
center_images = 1; % value of 1 will use the image allignment function to centre the images and fill space with average background
calculate_real_intensity = 1; % value of 1 will calculate the real intensity from the intensity file (example_setting == 2) - use if it has not already been done

% Define image settings
laser_intensity_scaling = 1; % scaling factor to change between pixel value and real laser intensity
pixel_size = 5.3*10^-6;
pixel_area = (pixel_size).^2; % 5.3 um by 5.3 um

% Defnie integration variables
integration_limits = [3*tau , -3*tau]; % [upper limit, lower limit]


%%%%  END OF USER INPUT %%%%


if example_setting == 0
    % Defining coordinates from intensity data - intensity values not important
    ReadOutIntensityStruct =doocsread('FLASH.DIAG/FFWDDIAG3.CAM/FF_FOCUS_SCAN/IMAGE');
    intensity_image = ReadOutIntensityStruct.data.d_byte_array_val;
    
    % creating coordinates and scales from the intensity image
    [size_x, size_y] = size(intensity_image);
    x_coor = 1:size_x;
    num_x = numel(x_coor);
    y_coor = 1:size_y;
    num_y = numel(y_coor);
    [X, Y] =meshgrid(x_coor, y_coor);
    
else % GAUSSIAN INTENSITY DISTROBUTION EXAMPLE
    % coordinates
    x_coor = -3.3973*10^-3:5.3*10^-6:3.3973*10^-3;  % in meters
    sig_x = 1;  % Gaussian width in x-axis
    c_x = 0;  % Displacement of gaussian from zero in x-axis
    num_x = numel(x_coor);
    
    y_coor = -2.7189*10^-3:5.3*10^-6:2.7189*10^-3;  % in meters
    sig_y = 1;  % Gaussian width in y-axis
    c_y = 0;  % Displacement of gaussian from zero in y-axis
    num_y = numel(y_coor);
    
end

z_data_slices = linspace(min_val,max_val,num_data_slices);

% Creating background image  - uncomment when creating new average
% background - see functions at end for details.
%background_image = creating_background_image(background_image_path,background_image_type);

background_image = importdata(background_image_path);

% Reallign the image and read intensity data at each distance step - This
% also runs the subroutines PicoMotorAllignment, XYProjection,
% PostProcessImageAlignment and plasma_density_calc.
[full_path_raw_image_data_array, plasma_density_ratio_path_array, Msquared_x, Msquared_y] = reading_intensity(x_coor,y_coor,z_data_slices,num_data_slices, laser_intensity_scaling, example_setting,raw_image_fig_path,raw_image_data_path,image_projection_path,raw_plasma_density_ratio_path,tau,Ki_i,ionisation_number,gas_density,integration_limits,hide_fig,sigma_plot_path, background_image, remove_background,calculate_plasma_density_ratio,center_images,pixel_size,pixel_area,only_take_data,step_size, Msquared_path, calculate_real_intensity, real_time_energy_path);

%Creating plots of data and saving
if only_take_data ~=1
% creation_of_plots(plasma_density_ratio_path_array,z_data_slices,num_data_slices, figure_path,raw_plasma_density_ratio_path,hide_fig,plasma_choice,step_size,full_plasma_ratio_data);
end
%%%%%%%% LOCAL FUNCTION DEFINITIONS %%%%%%%%

function [full_path_raw_image_data_array, plasma_density_ratio_path_array, Msquared_x, Msquared_y] = reading_intensity(x_coor,y_coor,z_data_slices,num_data_slices, laser_intensity_scaling, example_setting,raw_image_fig_path,raw_image_data_path,image_projection_path,raw_plasma_density_ratio_path,tau,Ki_i,ionisation_number,gas_density,integration_limits,hide_fig,sigma_plot_path,background_image, remove_background,calculate_plasma_density_ratio,center_images,pixel_size,pixel_area,only_take_data, step_size, Msquared_path, calculate_real_intensity, real_time_energy_path)

% reading intensity data at all desired z values with pico recalibration
% subscript. intensity_profile_data is a multidimensional array of
% all intensity profile readings. The function will now also save the image
% in a folder created dynamically with the date and time containing both
% the images and the images with the x/y projections overlayed.

full_path_raw_image_data_array = strings(num_data_slices,1);
plasma_density_ratio_path_array = strings(num_data_slices,1);
if example_setting == 0
    initial_position_stuct = doocsread('FLASH.DIAG/FFW.TUNNEL.FOCUS/MOTOR1/POS');
    initial_position = initial_position_stuct.data;
    for count_slice  = 1:num_data_slices
        setting_position =  doocswrite('FLASH.DIAG/FFW.TUNNEL.FOCUS/MOTOR1/POS.SET',z_data_slices(count_slice));
        delay_stage_actual_struct = doocsread('FLASH.DIAG/FFW.TUNNEL.FOCUS/MOTOR1/POS');
        delay_stage_actual = delay_stage_actual_struct.data;
        CMD_1 = doocswrite('FLASH.DIAG/FFW.TUNNEL.FOCUS/MOTOR1/CMD',1);
        
        difference = abs(delay_stage_actual - z_data_slices(count_slice));
        while difference > 2
            delay_stage_actual_struct = doocsread('FLASH.DIAG/FFW.TUNNEL.FOCUS/MOTOR1/POS');
            delay_stage_actual = delay_stage_actual_struct.data;
            difference = abs(delay_stage_actual - z_data_slices(count_slice));
            pause(1);
        end
        
        CMD_2 = doocswrite('FLASH.DIAG/FFW.TUNNEL.FOCUS/MOTOR1/CMD',2); %stopping the motor moving
        
        % Realligning the laser spot with pico motor after its driven by delay
        %stage using function
        PicoMotorAllignment();
        
        % Read the intensity data AFTER the pico motors have realligned the
        % image
        ReadOutIntensityStruct = doocsread('FLASH.DIAG/FFWDDIAG3.CAM/FF_FOCUS_SCAN/IMAGE');
        intensity_image = (ReadOutIntensityStruct.data.d_byte_array_val).*laser_intensity_scaling; % chnging from pixel value to intensity
        
        % Alligning the image to account for the pico motor tollerance
        if center_images ==1
        intensity_image = PostProcessImageAlignment(intensity_image,background_image);
        end
        
        % Calculating current z location
        z_location = z_data_slices(count_slice);
        
        % Calculating real intensity data from laser energy read out 
        intensity_profile_data = []; % allows intensity_from_energy_calc to be used for all cases
        [real_intensity_data] = intensity_from_energy_calc(intensity_image,intensity_profile_data,tau,pixel_area,integration_limits,real_time_energy_path,count_slice,z_location,example_setting);
        intensity_image = real_intensity_data;
        
        % Saving intensity image figure
        if hide_fig == 1
            intensity_image_fig = figure('visible', 'off'); %produces non-visible figure
        else
            intensity_image_fig = figure;
        end
        
        if remove_background ==1
        intensity_image = double(intensity_image) - background_image;
        intensity_image(intensity_image < 0) = 0;
        imagesc(intensity_image);
        else
        imagesc(intensity_image);  
        end
        
        raw_image_fig_part2 = sprintf('Raw_Image_SLICE_%d_Zaxis_%d__%s.fig',count_slice, z_location ,datestr(now,'HH:MM_dd-mm-yyyy'));
        full_path_raw_image_data = fullfile(raw_image_fig_path,raw_image_fig_part2);
        full_path_raw_image_data_array(count_slice,1) = string(full_path_raw_image_data);
        savefig(intensity_image_fig, full_path_raw_image_data);
        
        % Saving the raw intensity image data
        raw_image_data_part2 = sprintf('Raw_Image_Data_SLICE_%d_Zaxis_%d__%s.mat',count_slice, z_location ,datestr(now,'HH:MM_dd-mm-yyyy'));
        full_path_raw_image_data = fullfile(raw_image_data_path,raw_image_data_part2);
        save(full_path_raw_image_data,'intensity_image');
        full_path_raw_image_data_array(count_slice,1) = string(full_path_raw_image_data);
        if count_slice ==1
            number_rows_y =size(intensity_image,1);
            number_columns_x=size(intensity_image,2);
            x_summation = zeros(num_data_slices,number_columns_x);
            y_summation = zeros(number_rows_y,num_data_slices);
        end
        % calculating the x/y projections of the image as histograms
        if only_take_data ~= 1
            [x_summation, y_summation] = XYProjection(x_summation,y_summation,full_path_raw_image_data_array, count_slice,z_location, image_projection_path,hide_fig);
            
            % Calculation of plasma density from intensity profile data
            if calculate_plasma_density_ratio ==1
                plasma_density_ratio_path_array = plasma_density_calc(plasma_density_ratio_path_array,x_coor,y_coor,count_slice,z_location,tau,Ki_i, ionisation_number, gas_density, integration_limits,full_path_raw_image_data_array,raw_plasma_density_ratio_path,step_size);
            end
        end
    end
    
    %putting the delay stage back where it started and realliging picos
    initialposwrite =  doocswrite('FLASH.DIAG/FFW.TUNNEL.FOCUS/MOTOR1/POS.SET', initial_position);
    CMD_1 =  doocswrite('FLASH.DIAG/FFW.TUNNEL.FOCUS/MOTOR1/CMD', 1);
    difference = abs(delay_stage_actual - initial_position);
    while difference > 2
        delay_stage_actual_struct = doocsread('FLASH.DIAG/FFW.TUNNEL.FOCUS/MOTOR1/POS');
        delay_stage_actual = delay_stage_actual_struct.data;
        difference = abs(delay_stage_actual - z_data_slices(count_slice));
        pause(1);
    end
    
    % I think able to remove this bit of code for speed as the value is
    % measured again in PicoMotorAllignment 
%     ReadOutIntensityStruct =doocsread('FLASH.DIAG/FFWDDIAG3.CAM/FF_FOCUS_SCAN/IMAGE');
%     intensity_image = ReadOutIntensityStruct.data.d_byte_array_val;
%     %
    % Realligning the laser spot with pico motor after its driven by delay
    %stage
    PicoMotorAllignment();
    progress = fprintf('Data taken for slice %d out of %d total slices \n',count_slice,num_data_slices); 
elseif example_setting == 1
    laser_intensity = 12*10^13; % 7.5*10^18 W/cm^2 converted to 7.5*10^22 W/m^2 / 7.5*10^10um^2 laser intensity is taken from the flashforward design paper
    [X, Y] =meshgrid(x_coor, y_coor); %.*5.3*10^-6 to get from pixel to micrometer
    c_x = 0;
    c_y = 0;
    sig_x =1*10^-2;
    sig_y =1*10^-2;
    for count_slice = 1:num_data_slices   % EXAMPLE GAUSSIAN INTENSITY - NOT REAL DATA
        gauss_2D_pos(:,:,count_slice) = laser_intensity.*exp(-(((X-c_x).^2)./(2.*sig_x^2)+((Y-c_y).^2)./(2.*sig_y^2)));
        intensity_image = gauss_2D_pos(:,:,count_slice);
        
        if count_slice ==1
            number_rows_y =size(intensity_image,1);
            number_columns_x=size(intensity_image,2);
            x_summation = zeros(num_data_slices,number_columns_x);
            y_summation = zeros(number_rows_y,num_data_slices);
        end
        
        % Saving intensity image figure
        z_location = z_data_slices(count_slice);
        if hide_fig == 1
            intensity_image_fig = figure('visible', 'off'); %produces non-visible figure
        else
            intensity_image_fig = figure;
        end
        imagesc(intensity_image);
        raw_image_fig_part2 = sprintf('Raw_Image_SLICE_%d_Zaxis_%d__%s.fig',count_slice, z_location ,datestr(now,'HH:MM_dd-mm-yyyy'));
        full_path_raw_image_data = fullfile(raw_image_fig_path,raw_image_fig_part2);
        full_path_raw_image_data_array(count_slice,1) = string(full_path_raw_image_data);
        savefig(intensity_image_fig, full_path_raw_image_data);
        
        % Saving the intensity image data .mat file
        raw_image_data_part2 = sprintf('Raw_Image_SLICE_%d_Zaxis_%d__%s.mat',count_slice, z_location ,datestr(now,'HH:MM_dd-mm-yyyy'));
        full_path_raw_image_data = fullfile(raw_image_data_path,raw_image_data_part2);
        full_path_raw_image_data_array(count_slice,1) = string(full_path_raw_image_data);
        save(full_path_raw_image_data,'intensity_image');
        if only_take_data ~= 1
            [x_summation, y_summation] = XYProjection(x_summation, y_summation, full_path_raw_image_data_array, count_slice,z_location,image_projection_path,hide_fig);
            % Calculation of plasma density from intensity profile data
            if calculate_plasma_density_ratio ==1
          %      plasma_density_ratio_path_array = plasma_density_calc(plasma_density_ratio_path_array,x_coor,y_coor,count_slice,z_location,tau,Ki_i, ionisation_number, gas_density, integration_limits,full_path_raw_image_data_array,raw_plasma_density_ratio_path,step_size);
            end
        end
    end
    
    
elseif  example_setting == 2  %reads in data array ('intensity_profile_data') ordered by z position from lowest to highest. Saves these slices individually for analysis
    intensity_profile_data_struct =load('raw_plasma_data.mat', 'data_array');
    intensity_profile_data = intensity_profile_data_struct.data_array;
    intensity_image = []; count_slice =[]; z_location =[]; %so intensity_from_energy_calc can be used for different data type inputs (i.e single image or image array)
    if calculate_real_intensity == 1
        [real_intensity_data] = intensity_from_energy_calc(intensity_image,intensity_profile_data,tau,pixel_area,integration_limits,real_time_energy_path,count_slice, z_location,example_setting);
        intensity_profile_data = real_intensity_data;
    end
    z_location = [];%required so that the density function can be used for all three cases
    size_in_z = size(intensity_profile_data,3);
    z_data_slices = 1:num_data_slices;%size_in_z;
    number_rows_y =size(intensity_profile_data,1);
    number_columns_x=size(intensity_profile_data,2);
    x_summation = zeros(num_data_slices,number_columns_x);
    y_summation = zeros(number_rows_y,num_data_slices);
    for count_slice = 1:num_data_slices%size_in_z
        PREMEASURE_raw_image_fig_part2 = sprintf('PREMEASURED_Raw_Image_SLICE_%d.mat',count_slice);
        full_path_raw_image_data = fullfile(raw_image_data_path,PREMEASURE_raw_image_fig_part2);
        full_path_raw_image_data_array(count_slice,1) = string(full_path_raw_image_data);
        intensity_image = intensity_profile_data(:,:,count_slice);
        save(full_path_raw_image_data,'intensity_image');
        % Calculate and plot the x/y projection of the image
        if only_take_data ~= 1
            [x_summation, y_summation] =XYProjection(x_summation, y_summation, full_path_raw_image_data_array, count_slice,z_location,image_projection_path,hide_fig);
            % Calculation of plasma density from intensity profile data
            if calculate_plasma_density_ratio ==1
        %        plasma_density_ratio_path_array = plasma_density_calc(plasma_density_ratio_path_array,x_coor,y_coor,count_slice,z_location,tau,Ki_i, ionisation_number, gas_density, integration_limits,full_path_raw_image_data_array,raw_plasma_density_ratio_path,step_size);
            end
        end
    end
else
    error('Please Choose: example_setting = 0 for real data, example_setting = 1 for gaussian example, example_setting = 2 for loading data')
end

if only_take_data ~= 1
     Z_sigma_from_projections(x_summation, y_summation, z_data_slices, hide_fig, sigma_plot_path,step_size);
    [Msquared_x, Msquared_y] = calling_M2_functs(z_data_slices, raw_image_data_path, Msquared_path,step_size,pixel_size, hide_fig,example_setting)
end

end

function data_array = creating_intensity_image(data_path, data_type, example_setting)
data_full_path = fullfile(data_path,data_type);
data_struct = dir(data_full_path);

[N1] = fieldCount(data_struct);

all_data_names = strings(N1,1);
ordered_names = strings(N1,1);
for count = 1:N1
    all_data_names(count) = data_struct(count).name;
end

if example_setting ~=2
    for count_2 = 1:N1 %ordering the data by slice value in increasing order i.e 1,2,...N1
        location = find(contains(all_data_names,sprintf('Raw_Image_SLICE_%d_',count_2)));
        ordered_names(count_2) = all_data_names(location);
    end
end

if example_setting ==2
    for count_2 = 1:N1 %ordering the data by slice value in increasing order i.e 1,2,...N1
        location = find(contains(all_data_names,sprintf('PREMEASURED_Raw_Image_SLICE_%d.mat',count_2)));
        ordered_names(count_2) = all_data_names(location);
    end
end


for count_3 = 1:numel(ordered_names)
    data_array(:,:,count_3) = importdata(fullfile(char(data_path),char(ordered_names(count_3))));
end


end

function [Msquared_x, Msquared_y] = calling_M2_functs(z_data_slices, raw_image_data_path, Msquared_path,step_size,pixel_size,hide_fig,example_setting)
% NOTE : the wavelength is hard coded in the function M2_fit

z_data_slices = z_data_slices.*step_size;
data_path = raw_image_data_path;
data_type ='*.mat';
addpath(data_path) % this allows matlab to see the data regardless of the working directory
data_array = creating_intensity_image(data_path, data_type,example_setting);

[D4sigma_val_x, D4sigma_val_y] =  D4sigma(data_array,pixel_size);
[Msquared_x, Msquared_y, curve2, curve3, D4sigma_val_x, D4sigma_val_y] = M2_fit(D4sigma_val_x, D4sigma_val_y,z_data_slices);

% Plotting functions to visualise D4Sigma values for the laser across z
% values

if hide_fig == 1
    M2x_plot = figure('visible', 'off'); %produces non-visible figure
else
    M2x_plot = figure;
end
hold on
plot(z_data_slices,D4sigma_val_x)
title('x direction')
xlabel('z slices')
ylabel('D4\sigma Value')
plot(curve2)
hold off

if hide_fig == 1
    M2y_plot = figure('visible', 'off'); %produces non-visible figure
else
    M2y_plot = figure;
end

hold on
plot(z_data_slices,D4sigma_val_y)
title('y direction')
xlabel('z slices')
ylabel('D4\sigma Value')
plot(curve3)
hold off

% Saving plot
M2x_part2 = sprintf('M2x_plot__%s.fig',datestr(now,'HH:MM_dd-mm-yyyy'));
full_path_M2x = fullfile(Msquared_path,M2x_part2);
savefig(M2x_plot, full_path_M2x);

M2y_part2 = sprintf('M2y_plot__%s.fig',datestr(now,'HH:MM_dd-mm-yyyy'));
full_path_M2y = fullfile(Msquared_path,M2y_part2);
savefig(M2y_plot,full_path_M2y);

end

function [D4sigma_val_x, D4sigma_val_y] =  D4sigma(intensity_image,pixel_size)

% x direction calculation
sum_sum_image = sum(sum(intensity_image));
repeated_x = repmat(1:size(intensity_image,2),[size(intensity_image,1),1,size(intensity_image,3)]); %size(intensity_image,3) as last arg? 
x_bar = (sum(sum(intensity_image.*repeated_x)))./(sum_sum_image);
for count = 1:size(repeated_x,3)
coeff = (repeated_x(:,:,count)-x_bar(:,:,count)).^2;
coeff1(:,:,count)=coeff;
end
irradience_numerator_x = sum(sum(intensity_image.*(coeff1)));
irradience_denominator_x = sum_sum_image;
normalised_irradiance_x = irradience_numerator_x./irradience_denominator_x;
D4sigma_val_x = 4*sqrt(normalised_irradiance_x);
D4sigma_val_x = D4sigma_val_x.*pixel_size; %changing to real width rather than pixel width

clear x_bar repeated_x coeff1 coeff

% y direction calculation
repeated_y = repmat([1:size(intensity_image,1)]',[1,size(intensity_image,2),size(intensity_image,3)]);
y_bar = (sum(sum(intensity_image.*repeated_y)))./(sum_sum_image);
for count = 1:size(repeated_y,3)
coeff2 = (repeated_y(:,:,count)-y_bar(:,:,count)).^2;
coeff2y(:,:,count)=coeff2;
end
irradience_numerator_y = sum(sum(intensity_image.*(coeff2y)));
irradience_denominator_y = sum_sum_image;
normalised_irradiance_y = irradience_numerator_y./irradience_denominator_y;
D4sigma_val_y = 4*sqrt(normalised_irradiance_y);
D4sigma_val_y = D4sigma_val_y.*pixel_size; %changing to real width rather than pixel width
end

function [Msquared_fit_x, Msquared_fit_y , curve2, curve3, D4sigma_val_x, D4sigma_val_y] = M2_fit(D4sigma_val_x, D4sigma_val_y, z_data_slices) 
% Fits the variables to the data - necessary for calculating the w_0 and
% z_0 values of the beam data. 

%a = w_0, b = M^2, c = z_0 

%Fitting x data - known Z location, calculated D4sigma
% x_func = @(b,Z)(b(1).^2+b(2)^2.*(lamda./(pi.*b(1))).^2*(Z - b(3)).^2);

% Changing D4Sigma values to what is required for fitting func
D4sigma_val_x = (0.5*D4sigma_val_x).^2;
D4sigma_val_x(:) = D4sigma_val_x(1,1,:);
D4sigma_val_x = D4sigma_val_x(:);
%D4sigma_val_x = D4sigma_val_x';
D4sigma_val_y = (0.5*D4sigma_val_y).^2;
D4sigma_val_y(:) = D4sigma_val_y(1,1,:);
D4sigma_val_y = D4sigma_val_y(:);
% D4sigma_val_y = D4sigma_val_y';
% using 'fit' requires that the independent variable is stated as 'x'
x = z_data_slices';

% Fitting using D4Sigma_x
fit_op = fitoptions('Method','NonlinearLeastSquares','Lower',[0,1,0],'Upper',[Inf,Inf,max(x)],'StartPoint',[50, 100, 15]);
fit_ty = fittype('a.^2+b^2.*(800*10^-9./(pi.*a)).^2*(x - c).^2','options',fit_op);
[curve2,gof2] = fit(x,D4sigma_val_x,fit_ty);

w_0_x = curve2.a;
Msquared_fit_x = curve2.b;
Msquared_fit_x = sqrt(Msquared_fit_x);
z_0_x = curve2.c;

% Fitting using D4Sigma_y
fit_op = fitoptions('Method','NonlinearLeastSquares','Lower',[0,1,0],'Upper',[Inf,Inf,max(x)],'StartPoint',[50, 100, 15]);
fit_ty = fittype('a.^2+b^2.*(800*10^-9./(pi.*a)).^2*(x - c).^2','options',fit_op);
[curve3,gof2] = fit(x,D4sigma_val_x,fit_ty);

w_0_y = curve3.a;
Msquared_fit_y = curve3.b;
Msquared_fit_y = sqrt(Msquared_fit_y);
z_0_y = curve3.c;

end

function [real_intensity_data] = intensity_from_energy_calc(intensity_image,intensity_profile_data,tau,pixel_area,integration_limits,real_time_energy_path,count_slice, z_location,example_setting)
%This function takes the total energy incident on the sensor, normalises
%this sum of all pixel values to 1, then calculates the real intensity values at
%each pixel. 

upper_limit = integration_limits(1);
lower_limit = integration_limits(2);

% Reading real laser energy
real_time_energy = 800*10^-3; % rough example energy for testing in joules. 
% real_time_energy = doocsread('FLASH.DIAG/FFWDCONSTANTS/LASER_MP2_POWER/FLOAT_CONSTANT');

electric_field_time = @(t) exp(-1*(t.^2)./(2.*(tau.^2))); % creating the gaussian wavepacket
time_integrated = integral(electric_field_time,lower_limit,upper_limit); % integrating over wave packet

if example_setting ==2
    for count = 1:size(intensity_profile_data,3)
        sum_of_all_pixels = sum(sum(intensity_profile_data(:,:,count)));
        frac_of_energy = intensity_profile_data(:,:,count)./sum_of_all_pixels;
        energy_array = frac_of_energy.*real_time_energy;
        real_intensity_data(:,:,count) = energy_array./(time_integrated.*pixel_area*1281*1025);
    end
end

if example_setting ~=2
    real_time_file_name = sprintf('real_time_energy_slice_%d_Zaxis_%d__%s.mat',count_slice, z_location ,datestr(now,'HH:MM_dd-mm-yyyy'));
    real_teim_energy_path_full = fullfile(real_time_energy_path,real_time_file_name);
    save(real_teim_energy_path_full,'real_time_energy');

% converting the energy to power, then calculating the intensity by
% dividing by the area. 
sum_of_all_pixels = sum(sum(intensity_image));
frac_of_energy = intensity_image./sum_of_all_pixels;
energy_array = frac_of_energy.*real_time_energy;
real_intensity = energy_array./(time_integrated.*pixel_area);
real_intensity_data = real_intensity;
end

end

function PicoMotorAllignment()
%This function finds the centroid of the laser profile. It is based on
%whether or not a pixel is activated on the camera rather than the
%intensity of this pixel reading. The motor is then controlled in the
%appropriate axis to bring the laser profile back to the centre of the
%camera.

load background_image.mat;
ReadOutIntensityStruct =doocsread('FLASH.DIAG/FFWDDIAG3.CAM/FF_FOCUS_SCAN/IMAGE');
intensity_image = ReadOutIntensityStruct.data.d_byte_array_val;
intensity_image = double(intensity_image);
intensity_image = intensity_image - background_image;
intensity_image(intensity_image<0)=0;

pico1_const_horz = 1.3; %controls the step size that pico 1 takes in horizontal direction
pico1_const_vert = 1.3; %controls the step size that pico 1 takes in vertical direction

% calculation of centroid (the comparison will depend on whether the matrix is centred on zero or zero is in the bottom left)
[size_x , size_y] = size(intensity_image);
x_centre = size_y./2; %set the x position of the centre of the camera
y_centre = size_x./2;%set the y position of the centre of the camera
x_sum=0;
y_sum=0;
count_x=0;
count_y=0;
tolerance = 20;  % 10 is arbitrary at the moment - perfect allignment would have the centroid at (0,0) if this is the index for the centre of the camera
x_mean = 100; %these values set so that the while loop is satisfied for at aleast 1 iteration
y_mean = 100; %these values set so that the while loop is satisfied for at aleast 1 iteration

sum_x_centroid = sum(intensity_image,1);
sum_y_centroid = sum(intensity_image,2);
x_mean = sum((sum_x_centroid.*[1:length(sum_x_centroid)])./(sum(sum_x_centroid)));
y_mean = sum((sum_y_centroid.*[1:length(sum_y_centroid)]')./(sum(sum_y_centroid)));

weighted_position_x = abs(x_mean - x_centre);
weighted_position_y = abs(y_mean - y_centre);


% intensity_image(intensity_image < 5) = 0; % 5 is arbitrary, removing the background with logical indexing, possibly not necessary?
%
while (weighted_position_x > tolerance) || (weighted_position_y > tolerance )   
   load background_image.mat;
    ReadOutIntensityStruct =doocsread('FLASH.DIAG/FFWDDIAG3.CAM/FF_FOCUS_SCAN/IMAGE');
    intensity_image = ReadOutIntensityStruct.data.d_byte_array_val;
    intensity_image = double(intensity_image);
    intensity_image = intensity_image - background_image;
    intensity_image(intensity_image<0)=0;
    
    % new centroid calculation - x
    sum_x_centroid = sum(intensity_image,1);
    x_mean = sum((sum_x_centroid.*[1:length(sum_x_centroid)])./(sum(sum_x_centroid)));
    
    intensity_centroid = [x_mean,y_mean];
    weighted_position_x = abs(x_mean - x_centre);

    
    
   
    % VERTICAL PICO MOTOR
    
    if x_mean > (x_centre + tolerance) %this may need to be the centre pixel number instead
        set_position_vert = weighted_position_x.*pico1_const_vert; % moves pico motor vertically in negative direction
        pico_1_vert_set = doocswrite('FLASH.DIAG/FFWD.PICO.5/PICO_DEVICE1/PICOMOTOR.TARGET_POSITION1', set_position_vert);   %pico 1 - vertical
        pause(1)
        pico_1_vert_move = doocswrite('FLASH.DIAG/FFWD.PICO.5/PICO_DEVICE1/PICOMOTOR.COMMAND1', 1); %pico 1, vertically, move command
    elseif x_mean < (x_centre - tolerance)
        set_position_vert  = -1*weighted_position_x.*pico1_const_vert ;% moves pico motor vertically in positive direction
        pico_1_vert_set = doocswrite('FLASH.DIAG/FFWD.PICO.5/PICO_DEVICE1/PICOMOTOR.TARGET_POSITION1', set_position_vert);  %pico 1 - vertical
        pause(1)
        pico_1_vert_move = doocswrite('FLASH.DIAG/FFWD.PICO.5/PICO_DEVICE1/PICOMOTOR.COMMAND1', 1); %pico 1, vertical, move command
    end
    pause(2)
    
     % HORIZONTAL PICO MOTOR
    
     % new centroid calculation - y
    sum_y_centroid = sum(intensity_image,2);
    y_mean = sum((sum_y_centroid.*[1:length(sum_y_centroid)]')./(sum(sum_y_centroid)));
    weighted_position_y = abs(y_mean - y_centre);
    
    if y_mean > (y_centre + tolerance) %this may need to be the centre pixel number instead
        set_position_horz = weighted_position_y.*pico1_const_horz; % moves pico moto horizontally in negative direction
        pico_1_horz_set = doocswrite('FLASH.DIAG/FFWD.PICO.5/PICO_DEVICE1/PICOMOTOR.TARGET_POSITION2', set_position_horz);  %pico 1 - horizontal
        pause(1)
        pico_1_horz_move = doocswrite('FLASH.DIAG/FFWD.PICO.5/PICO_DEVICE1/PICOMOTOR.COMMAND2', 1); %pico 1, horizontal, move command
        pause(2)
    elseif y_mean < (y_centre - tolerance)
        set_position_horz =-1*weighted_position_y.*pico1_const_horz; % moves pico motor horizontally in positive direction
        pico_1_horz_set = doocswrite('FLASH.DIAG/FFWD.PICO.5/PICO_DEVICE1/PICOMOTOR.TARGET_POSITION2', set_position_horz);  %pico 1 - horizontal
       pause(1)
        pico_1_horz_move = doocswrite('FLASH.DIAG/FFWD.PICO.5/PICO_DEVICE1/PICOMOTOR.COMMAND2', 1); %pico 1, horizontal, move command
  pause(2)
    end
    
end


end

function Alligned_image = PostProcessImageAlignment(intensity_image,background_image)
% This function takes the imaged poduced by the measurement of the delay
% stage camera after the pico motor allignment and fine tunes the position
% of the centre of the image. This is necessary due to the tolerance of the
% pico motor position subroutine. The blank pixels due to moving the image
% are filled with average background of the camera.

intensity_image = double(intensity_image);
initial_dim_row = size(intensity_image,1);
initial_dim_col = size(intensity_image,2);

col_sum_align = sum(intensity_image,1);
row_sum_align = sum(intensity_image,2);

col_weighted_center = find(col_sum_align==median(col_sum_align(:)));
row_weighted_center = find(row_sum_align==median(row_sum_align(:)));

A=mat2gray(double(intensity_image));
center_pixel=size(A)/2+.5;
move_col = center_pixel(2)-col_weighted_center; %will move negative for an apparent image centre larger than actual image centre and vice versa
move_row = center_pixel(1)-row_weighted_center;
Alligned_image = imtranslate(intensity_image,[move_col, move_row]);

if move_col < 0
    lower_col = initial_dim_col + move_col;
    fill_vec_col = lower_col:initial_dim_col;
elseif move_col > 0
    fill_vec_col = 1:move_col;
end

if move_row < 0
    lower_row = initial_dim_row + move_row;
    fill_vec_row = [lower_row:initial_dim_row];
elseif move_row >0
    fill_vec_row = [1:move_row];
end

background_reduced_col = background_image(:,fill_vec_col);
background_reduced_row = background_image(fill_vec_row,:);
Alligned_image(:,fill_vec_col)=background_reduced_col;
Alligned_image(fill_vec_row,:)=background_reduced_row;

end

function [x_summation, y_summation] = XYProjection(x_summation,y_summation,full_path_raw_image_data_array, count_slice,z_location, image_projection_path,hide_fig)

%loading saved intensity data for given count_slice value
intensity_data_path_to_load = char(full_path_raw_image_data_array(count_slice,1));
intensity_image_struct = load(intensity_data_path_to_load);
intensity_image = intensity_image_struct.intensity_image;

%data for reading values
xwidth = 1:size(intensity_image, 2);
xcount = sum(intensity_image, 1);
yheight = 1:size(intensity_image, 1);
ycount = sum(intensity_image, 2);

%scaled image for comparison
[x_size,y_size] = size(intensity_image);
y_vec = 1:x_size;
x_vec = 1:y_size;
x_summation(count_slice,:) = sum(intensity_image,1);
y_summation(:,count_slice) = sum(intensity_image,2);
x_summation_scaled(count_slice,:) = mat2gray(x_summation(count_slice,:)).*100; %creates a grey scale image with values between [0,1]
y_summation_scaled(:,count_slice) = mat2gray(y_summation(:,count_slice)).*100;

if hide_fig == 1
    xy_Projection_Fig = figure('visible', 'off'); %produces non-visible figure
else
    xy_Projection_Fig = figure;
end

subplot(2, 2, 2);
bar(xwidth, xcount,'g');
xlabel('Pixel Position Along X');
ylabel('Pixel Value Count');
title('X Projection Histogram');
xlim([0 max(x_vec)]);
subplot(2, 2, 1);
barh(yheight, ycount,'r');
xlabel('Pixel Value Count');
ylabel('Pixel Position Along Y');
title('Y Projection Histogram');
ylim([0 max(y_vec)]);
AX = subplot(2, 1, 2);
hold on
imagesc(intensity_image);
plot(y_summation_scaled(:,count_slice), y_vec,'Color','r','LineWidth',2);
plot(x_vec,x_summation_scaled(count_slice,:),'Color','g','LineWidth',2);
hold off
xlim([0 max(x_vec)]);
ylim([0 max(y_vec)]);
legend('X-axis Projection','Y-axis Projection','Location','northeast')
oldTickY = get(AX,'YTick'); %get the current tick points of the y axis
oldTickX = get(AX,'XTick');
newTickYStr = cellstr(num2str(oldTickY'*5.3)); %create a cell array of strings
newTickXStr = cellstr(num2str(oldTickX'*5.3)); %create a cell array of strings
set(AX,'YTickLabel',newTickYStr, 'XTickLabel', newTickXStr);
xlabel('X Spacial Dimension \mum');
ylabel('Y Spacial Dimension \mum');

xy_filepath = image_projection_path;
xy_filename = sprintf('XY_projection_slice_%d_Zaxis_%d__%s.fig',count_slice, z_location ,datestr(now,'HH:MM_dd-mm-yyyy'));
xy_filepath_full = fullfile(xy_filepath,xy_filename);
savefig(xy_Projection_Fig,xy_filepath_full,'compact');
end

function Z_sigma_from_projections(x_summation, y_summation,z_data_slices, hide_fig, sigma_plot_path,step_size)
depth_xsum = size(x_summation,1);
depth_ysum = size(y_summation,2);

z_data_slices = z_data_slices.*step_size;

for sig_count = 1:depth_xsum
    sig_x(sig_count) = std(x_summation(sig_count,:))./mean(x_summation(sig_count,:));
    sig_y(sig_count) = std(y_summation(:,sig_count))./mean(y_summation(:,sig_count));
end

%if hide_fig == 1
 %   sigma_fig= figure('visible', 'off'); %produces non-visible figure
%else
    sigma_fig = figure;
%end

%fitting curves
p_x = polyfit(z_data_slices,sig_x,1);
p_y = polyfit(z_data_slices,sig_y,1);
x_fit = polyval(p_x,z_data_slices);
y_fit = polyval(p_y,z_data_slices);

%plotting
hold on
plot(z_data_slices,x_fit);
plot(z_data_slices,y_fit);
s_x = scatter(z_data_slices,sig_x,'x','r');
s_y = scatter(z_data_slices,sig_y,'+','b');
hold off
% s_x.LineWidth = 0.6;
% s_x.MarkerEdgeColor = 'b';
% s_x.MarkerFaceColor = [0 0.5 0.5];

% s_x.LineWidth = 0.6;
% s_x.MarkerEdgeColor = 'b';
% s_x.MarkerFaceColor = [0 0.5 0.5];

z_x = xlabel('Z Spacial Dimension \mum');
z_y = ylabel('\sigma_x , \sigma_y');
z_l = legend('Standard Deviation in X Projection','Standard Deviation in Y Projection','Location','east');
% Saving plot
sigma_part2 = sprintf('Sigma_plot__%s.fig',datestr(now,'HH:MM_dd-mm-yyyy'));
full_path_sigma = fullfile(sigma_plot_path,sigma_part2);
savefig(sigma_fig, full_path_sigma);

end

function plasma_density_ratio_path_array = plasma_density_calc(plasma_density_ratio_path_array,x_coor,y_coor,count_slice,z_location,tau,Ki_i, ionisation_number,gas_density, integration_limits,full_path_raw_image_data_array,raw_plasma_density_ratio_path,intensity_profile_data)
% Calculates the plasma density ratio from the intensity profile data

intensity_data_path_to_load = char(full_path_raw_image_data_array(count_slice,1));
intensity_image_struct = load(intensity_data_path_to_load);
intensity_image = intensity_image_struct.intensity_image;


% Defining symbols and predefining arrays for speed
size_intensity = size(intensity_image);

% Defining physical variables
permitivity = 8.85418782*10^(-12); %for vaccuum
c = 3*10^8;

% Collecting integration variables
upper_limit = integration_limits(1);
lower_limit = integration_limits(2);

% Calculation of electric field from intensity
electric_field = sqrt((intensity_image.*2)./(permitivity*c))./10^9; %electric field from intensity - rp photonics website, 10^9 to get it into GV
% The expected electric field should be between roughly 1 and 1000 GV - NOT ELECTRON VOLTS!

% Temporal variation of electric field - modelled as gaussian - 't' is time
% variable
electric_field_time = @(t) electric_field.*exp(-1*(t.^2)./(2.*(tau.^2)));

% Calculation of parameters based on plasma characteritics
eff_principle_n = ionisation_number/sqrt(2*Ki_i);
gamma_n = gamma(2*eff_principle_n);

% Calculation of the field value at which TI is valid
crit_electric_field = (sqrt(2)-1)*abs(Ki_i)^(3/2)*5.14*10^11; %can be used for triggering different ionsiation modelling method

% Calculation of ADK rate
term_1 = 1.52*10^15;
frac_1 = (4.^eff_principle_n*Ki_i)./(eff_principle_n*gamma_n);
frac_2 = @(t) (20.5.*Ki_i.^(3/2)./electric_field_time(t)).^(2.*eff_principle_n-1);
frac_3 = @(t) exp((-6.83.*(Ki_i.^(3/2)))./(electric_field_time(t)));

w = @(t) term_1.*frac_1.*frac_2(t).*frac_3(t);% ADK ionisation rate (s^-1)
% sym_w=sym(w);
int_w = integral(w,lower_limit,upper_limit,'ArrayValued',true); % Integrating over the laser pulse
%int_w(isnan(int_w))=0; % removing NaNs after the integration - these are caused by zero intensities after removing backgrounds
plasma_density_ratio = gas_density*(1 - exp(-1*int_w)); % calculation from Alex thesis

raw_plasma_data_part2 = sprintf('Raw_Plasma_Data_SLICE_%d_Zaxis_%d__%s.mat',count_slice, z_location ,datestr(now,'HH:MM_dd-mm-yyyy'));
full_path_raw_plasma_data = fullfile(raw_plasma_density_ratio_path,raw_plasma_data_part2);
plasma_density_ratio_path_array(count_slice,1) = string(full_path_raw_plasma_data);
save(full_path_raw_plasma_data,'plasma_density_ratio');

end

function creation_of_plots(plasma_density_ratio_path_array,z_data_slices,num_data_slices, figure_path,raw_plasma_density_ratio_path,hide_fig,plasma_choice,step_size,full_plasma_ratio_data)

% create plots in slices with real z distance, combines them and saves to
% the specified directory with a time dependent file name. Controls if the
% plotting is hidden or not. This loads ALL plasma ratio data to create
% plots
z_data_slices = z_data_slices.*step_size;
% loading required plasma density ratio data
for slice = 1:num_data_slices
    plasma_data_path_to_load = char(plasma_density_ratio_path_array(slice,1));
    plasma_data_struct = load(plasma_data_path_to_load);
    plasma_density_ratio(:,:,slice) = plasma_data_struct.plasma_density_ratio;
end

% Controlling the suppression of the figure
if hide_fig == 1
    FIG = figure('Name','Plasma Density Plot','NumberTitle','off', 'visible', 'off'); %produces non-visible figure
else
    FIG = figure('Name','Plasma Density Plot','NumberTitle','off');
end

hold on
for k = 1:num_data_slices
    [~,h] = contourf(plasma_density_ratio(:,:,k),20);  %plots 20 contours for the plasma density ratio for slice 'k' in z_data_slices
    h.ContourZLevel = z_data_slices(k);  % moves the 2D contour to the correct z location on plot
end
hold off
colorbar
xlabel('Spacial X Dimension')
ylabel('Spacial Y Dimension')
zlabel('Z Motor Position Values')
h = colorbar;
if plasma_choice == 1
    ylabel(h, 'Plasma Density Ratio, ^{n_e}/_{n_0}')
else
    ylabel(h, 'Absolute Plasma Density, n_e')
end

view(3); axis vis3d; grid on

% Save the plots with the current time stamp as the name and a subdirectory
figpart2 = sprintf('Plasma_density_%s.fig', datestr(now,'HH:MM_dd-mm-yyyy'));
f = fullfile(figure_path, figpart2);
savefig(FIG,f,'compact');
raw_data_part2 = sprintf('Raw_data_Plasma_density_%s.mat', datestr(now,'HH:MM_dd-mm-yyyy'));
g = fullfile(full_plasma_ratio_data,raw_data_part2);
save(g,'plasma_density_ratio');
end

%  USE THE FOLLOWING FUNCTIONS FOR CREATING A BACKGROUND IF IMAGES ARE RETAKEN.
%  MAKE SURE YOU ARE IN THE DIRECTORY OF THE IMAGES FOR SUCCESSFUL
%  BACKGROUND CREATION.


function background_image = creating_background_image(background_image_path, background_image_type)
 background_files_full_path = fullfile(background_image_path,background_image_type);
 background_files_struct = dir(background_files_full_path);

[N1] = fieldCount(background_files_struct);

background_files_names = strings(N1,1);
for count = 1:N1
background_files_names(count) = background_files_struct(count).name;
end

for count_2 = 1:numel(background_files_names)
   background_data(:,:,count_2) = importdata(char(background_files_names(count_2)));

end

background_image = ((sum(background_data,3))./N1)';
end

function [N1] = fieldCount(inputStruct)  % function to count the number of entries in the struct catogery 'name'
N1 = 0;
for i = 1: numel(inputStruct)
  if(~isempty(getfield(inputStruct(i),'name')))  % gets the array 'name'from the struct
      N1 = N1 + 1;
  end
end
end