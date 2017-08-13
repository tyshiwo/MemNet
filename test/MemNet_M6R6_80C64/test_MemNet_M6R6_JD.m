%% --------------------------
% MemNet_M6R6 for JPEG deblocking
% edit by yingtai 12/08/2017
% -------------------------------
function test_MemNet_M6R6_JD()
setenv('LC_ALL','C')
addpath /data2/taiying/MSU_Code/119-caffe-matlab/matlab; % change to your caffe path
setenv('GLOG_minloglevel','2')
addpath('../');
addpath('../evaluation_func/');
addpath('../evaluation_func/matlabPyrTools-master/');

%% parameters
gpu_id = 7;
JPEG_Quality = 20;
data_set_id = 1;
thresh_hei = 150; % threshold patch size for inference, since too big image may cost too much memory
thresh_wid = 150;
rf = 16;

pathfolder = ['../../data/JPEGDeblocking/'];
if data_set_id == 1
    % classic5
    setTestCur = 'classic5';
    path = [pathfolder setTestCur '/'];
    d = dir([path '*.bmp']);
    filenum = 5;
end
if data_set_id == 2
    % LIVE1
    setTestCur = 'LIVE1';
    path = [pathfolder setTestCur '/'];
    d = dir([path '*.bmp']);
    filenum = 29;
end

savepath = ['./results/'];
folderResultCur = fullfile(savepath, [setTestCur,'_Quality',num2str(JPEG_Quality)]);
%%% folder to store results
if ~exist(folderResultCur,'file')
    mkdir(folderResultCur);
end

mean_compressed = [];
mean_memnet = [];
% caffe.set_mode_cpu(); % for CPU
caffe.set_mode_gpu(); % for GPU
caffe.set_device(gpu_id);

weights = ['../../model/MemNet_M6R6_80C64_JD.caffemodel'];
model_path = './MemNet_M6R6_80C64_deploy';

compressed_set =[];
memnet_set = [];
im_b_set = cell(filenum,1);
im_h_set = cell(filenum,1);
im_gnd_set = cell(filenum,1);

for iii = 1:1:length(d)
    disp(['id: ' num2str(iii)]);
    imageName = d(iii).name;
    imageName = imageName(1:end-4);
    im  = imread([path d(iii).name]);
    
    %% work on luminance only
    if size(im,3)>1
        im = rgb2ycbcr(im);
    end
    im_y = im(:,:,1);
    im_gnd = im2double(im_y);
    [hei,wid,channels] = size(im_gnd);
    
    %% generate JPEG-compressed input
    imwrite(im_gnd,'../im_JPEG.jpg','jpg','Quality',JPEG_Quality);
    im_b = im2double(imread('../im_JPEG.jpg'));
    im_b = single(im_b);
    
    %% adaptively spilt
    % decide patch numbers
    hei_patch = ceil(hei/(thresh_hei+rf));
    wid_patch = ceil(wid/(thresh_wid+rf));
    hei_stride = ceil(hei/hei_patch);
    wid_stride = ceil(wid/wid_patch);
    use_start_x = 0;
    use_start_y = 0;
    use_end_x = 0;
    use_end_y = 0;
    
    ext_start_x = 0;
    ext_end_x = 0;
    ext_start_y = 0;
    ext_end_y = 0;
    
    posext_start_x = 0;
    posext_start_y = 0;
    posext_end_x = 0;
    posext_end_y = 0;
    
    % extract each patch for inference
    im_h = [];
    for x = 1 : hei_stride : hei
        for y = 1 : wid_stride : wid
            % decide the length of hei and wid for each patch
            use_start_x = x;
            use_start_y = y;
            if x - rf > 1 % add border
                ext_start_x = x-rf;
                posext_start_x = rf+1;
            else
                ext_start_x = x;
                posext_start_x = 1;
            end
            if y-rf > 1
                ext_start_y = y-rf;
                posext_start_y = rf+1;
            else
                ext_start_y = y;
                posext_start_y = 1;
            end
            
            use_end_x = use_start_x+hei_stride-1;
            use_end_y = use_start_y+wid_stride-1;
            
            
            if use_start_x+hei_stride+rf-1 <= hei
                hei_length = hei_stride+rf;
                ext_end_x = use_start_x+hei_length-1;
                posext_end_x = hei_length-rf+posext_start_x-1;
                
            else
                hei_length = hei-ext_start_x+1;
                ext_end_x = ext_start_x+hei_length-1;
                posext_end_x = hei_length;
                use_end_x = ext_start_x+hei_length-1;
            end
            if use_start_y+wid_stride+rf-1 <= wid
                wid_length = wid_stride+rf;
                ext_end_y = use_start_y+wid_length-1;
                posext_end_y = wid_length-rf+posext_start_y-1;
                
            else
                wid_length = wid-ext_start_y+1;
                ext_end_y = ext_start_y+wid_length-1;
                posext_end_y = wid_length;
                use_end_y = ext_start_y+wid_length-1;
            end
            
            subim_input = im_b(ext_start_x : ext_end_x, ext_start_y : ext_end_y);  % input
            data = permute(subim_input,[2, 1, 3]);
            model = [model_path '.prototxt'];
            subim_output = do_cnn(model,weights,data);
            subim_output = subim_output';
            subim_output = subim_output(posext_start_x:posext_end_x,posext_start_y:posext_end_y);
            % fill im_h with sub_output
            im_h(use_start_x:use_end_x,use_start_y:use_end_y) = subim_output;
        end
    end
    
    im_b1 = (single(im_b) * 255);
    im_h1 = (single(im_h) * 255);
    im_gnd1 = (single(im_gnd) * 255);
    
    im_b_set{iii} = im_b1;
    im_h_set{iii} = im_h1;
    im_gnd_set{iii} = im_gnd1;
    
    %% compute PSNR and SSIM and IFC
    compressed(1) = compute_psnr(im_gnd1,im_b1);
    memnet(1) = compute_psnr(im_gnd1,im_h1);
    compressed(2) = ssim_index(im_gnd1,im_b1);
    memnet(2) = ssim_index(im_gnd1,im_h1);
    
    compressed_set = [compressed_set; compressed];
    memnet_set = [memnet_set; memnet];
    %% save images
    imwrite(uint8(im_h1),fullfile(folderResultCur,[imageName,'_Quality',num2str(JPEG_Quality),'.png']));
end
mean_compressed = [mean_compressed; [mean(compressed_set(:,1)) mean(compressed_set(:,2))]];
mean_memnet = [mean_memnet; [mean(memnet_set(:,1)) mean(memnet_set(:,2))]];


%%% save PSNR and SSIM metrics
PSNR_set = memnet_set(:,1);
SSIM_set = memnet_set(:,2);
save(fullfile(folderResultCur,['PSNR_',setTestCur,'_Quality',num2str(JPEG_Quality),'.mat']),['PSNR_set'])
save(fullfile(folderResultCur,['SSIM_',setTestCur,'_Quality',num2str(JPEG_Quality),'.mat']),['SSIM_set'])

disp(['compressed = ' num2str(mean_compressed(1,:)) '---- MemNet = ' num2str(mean_memnet(1,:))]);

end
 


