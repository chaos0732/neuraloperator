import com.comsol.model.*
import com.comsol.model.util.*

% 加载模型文件
model = mphload('D:\MyCodes\pythoncodes\NeuralOperator\My_data\metaunit_PML_Ex.mph');

% 设置输出路径
outputFolder = 'D:\MyCodes\pythoncodes\NeuralOperator\My_data\unit_64_multisample';
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% 控制参数 sweep（例如26种结构）
for i = 1:26
    % 设置结构参数 w
    w_val = 2 + 0.1 * (i - 1);  % 从 2 ~ 4.5
    model.param.set('w', sprintf('%.4f', w_val));
    
    % 运行仿真
    model.study('std1').run();
    exportTag1 = sprintf('fielddata_%d', i);
    try
        model.result.export.remove(exportTag1);  % ✅ 正确写法
    catch
    end

    % ============== ① 导出 z = 6 mm 面上的 Ex 实部 + Ey 虚部 ==============
    
    % 创建 "Data" 类型的导出节点
    exportNode1 = model.result.export.create(exportTag1, 'Data');
    
    % 1. 选择数据源(可选)
    % 若只有一个研究或一个解，可以用默认，也可指定：
    % exportNode1.set('data', 'solution'); 
    % 如果需要体数据或表面数据，可再调整 'data'、'dataset' 等。

    % 2. 设置需要输出的表达式
    exportNode1.set('expr', {'real(emw.Ex)', 'imag(emw.Ey)'});
    exportNode1.set('unit', {'V/m', 'V/m'});
    
    % 3. 网格采样相关
    exportNode1.set('grid', 'on');            % 打开网格导出
    exportNode1.set('gridxnumber', 64);
    exportNode1.set('gridynumber', 64);
    exportNode1.set('gridxmin', '0');         % X 范围可根据实际模型设置
    exportNode1.set('gridxmax', '5');         % 假设结构是 0~5 mm
    exportNode1.set('gridymin', '0');
    exportNode1.set('gridymax', '5');
    exportNode1.set('gridz', '6[mm]');        % z=6 mm 截面

    % 4. 设置输出文件
    file_id = sprintf('%02d', i);  % 自动编号
    exportNode1.set('filename', fullfile(outputFolder, ['Ex_' file_id '.txt']));
    
    % 5. 执行导出
    exportNode1.run;

    % ============== ② 导出 z = 4 mm 的 dom 掩膜图样 ==============
    exportTag2 = sprintf('fielddata_%d', i);
    try
        model.result.export.remoe(exportTag2);  % 正确写法
    catch
    end

    exportNode2 = model.result.export.create(exportTag2, 'Data');
    % 1. 设置表达式
    exportNode2.set('expr', {'dom'});  % 请确保 'dom' 是你在模型中定义的变量名

    % 2. 网格采样设置
    exportNode2.set('grid', 'on');
    exportNode2.set('gridxnumber', 64);
    exportNode2.set('gridynumber', 64);
    exportNode2.set('gridxmin', '0');
    exportNode2.set('gridxmax', '5');
    exportNode2.set('gridymin', '0');
    exportNode2.set('gridymax', '5');
    exportNode2.set('gridz', '4[mm]');  % z=4 mm

    % 3. 设置输出文件
    exportNode2.set('filename', fullfile(outputFolder, ['input_' file_id '.txt']));
    
    % 4. 执行导出
    exportNode2.run;
end

disp('✅ 所有结构数据导出完毕！');
