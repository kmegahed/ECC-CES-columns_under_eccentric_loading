function [force, check] = model(X)
    % Extract columns from input matrix X
    B1 = X(:,1); H1 = X(:,2); lam1 = X(:,3); fy_rebar1 = X(:,4);
    fc1 = X(:,5); fs_steel1 = X(:,6); d_steel1 = X(:,7); ro_rebar1 = X(:,8);
    tf_tw1 = X(:,9); cH1 = X(:,10); cB1 = X(:,11);eeee1 = X(:,12);
    
    force = [];
    check = [];
    
    outputDirectory = 'C:\Users\osama\OneDrive\Desktop\abaqus2\o\';  % Output directory
    
    for i = 1:numel(B1)
        % Extract and round values
        B = B1(i);    H = H1(i);    lam = lam1(i);    fy_rebar = fy_rebar1(i);    fc = fc1(i);    fs_steel = fs_steel1(i);
        d_steel = d_steel1(i);    ro_rebar = ro_rebar1(i);    tf_tw2 = tf_tw1(i);    cH = cH1(i);    cB = cB1(i);ee=eeee1(i);eff=exp(ee)-0.9;ec=(eff)*H;
        new_jjj = sprintf('%d_%d_%d_%d', round(B), round(H), round(lam), round(eff*100));%'%d_%0.2f_%0.1f_%d'
        % Modify and write the first script
        inputFilePath = 'C:\Users\osama\OneDrive\Desktop\abaqus2\script_m.py';
        outputFilePath = fullfile(outputDirectory, [new_jjj '.py']);
        
        % Read, modify, and write script content
        fid = fopen(inputFilePath, 'r');
        scriptContent = fread(fid, '*char').';
        fclose(fid);
        
        modifiedContent = strrep(scriptContent, 'B=300.0', sprintf('B=%d', B));
        modifiedContent = strrep(modifiedContent, 'H=600.0', sprintf('H=%d', H));
        modifiedContent = strrep(modifiedContent, 'lam=5.0', sprintf('lam=%d', lam));
        modifiedContent = strrep(modifiedContent, 'fr=250.0', sprintf('fr=%d', fy_rebar));
        modifiedContent = strrep(modifiedContent, 'fe=25.0', sprintf('fe=%d', fc));
        modifiedContent = strrep(modifiedContent, 'fysteel=230.0', sprintf('fysteel=%d', fs_steel));
        modifiedContent = strrep(modifiedContent, 'd_steel=0.3', sprintf('d_steel=%d', d_steel));
        modifiedContent = strrep(modifiedContent, 'ro_rebar=0.01', sprintf('ro_rebar=%d', ro_rebar));
        modifiedContent = strrep(modifiedContent, 'tf_tw=1.50', sprintf('tf_tw=%d', tf_tw2));
        modifiedContent = strrep(modifiedContent, 'cH=60', sprintf('cH=%d', cH));
        modifiedContent = strrep(modifiedContent, 'cB=50', sprintf('cB=%d', cB));
        modifiedContent = strrep(modifiedContent, 'eccXY=600', sprintf('eccXY=%d', ec));
        modifiedContent = strrep(modifiedContent, 'mod1=''Model-1''', sprintf('mod1=''%s''', new_jjj));
        
        fid = fopen(outputFilePath, 'w');
        fwrite(fid, modifiedContent);
        fclose(fid);
        
        % Run the first script
        system(['abaqus cae noGUI=' outputFilePath]);
        %pause(20);
        % Modify and run the second script
        inputFilePath2 = 'C:\Users\osama\OneDrive\Desktop\abaqus2\script2.py';
        
        fid = fopen(inputFilePath2, 'r');
        scriptContent = fread(fid, '*char').';
        fclose(fid);
        
        modifiedContent = strrep(scriptContent, 'mmmooo', new_jjj);
        modifiedContent = strrep(modifiedContent, 'abaqusoo', new_jjj);
        
        outputFilePath2 = fullfile(outputDirectory, [new_jjj '_2.py']);
        fid = fopen(outputFilePath2, 'w');
        fwrite(fid, modifiedContent);
        fclose(fid);
        
        system(['abaqus cae noGUI=' outputFilePath2]);
        
        % Read the output .rpt file
        rptFilePath = fullfile(outputDirectory, [new_jjj '.rpt']);
        
        fid = fopen(rptFilePath, 'r');
        data = textscan(fid, '%f %f', 'HeaderLines', 2);
        fclose(fid);
        
        % Extract data and calculate force
        X = data{1};
        aa = data{2} * -1 / 1000.0;
        
        forcei = max(aa);
        checki = aa(end) < 0.995 * forcei;
        
        % Modify the script if checki is false
        if ~checki
            disp('jjjjjjjjjjj')
            %outputFilePath is file.py (for run again)
            fid = fopen(outputFilePath, 'r');   scriptContent = fread(fid, '*char').'; fclose(fid);
            % change only u3 in the file.py
            modifiedContent = strrep(scriptContent, 'u3=L/', 'u3=L/1*1.7/');
            %write content modified
            fid = fopen(outputFilePath, 'w');fwrite(fid, modifiedContent);fclose(fid);
            %run the abaqus simulation
            system(['abaqus cae noGUI=' outputFilePath]);

            %inputFilePath2 is script2.py (visulaize)
            %open read file
            fid = fopen(inputFilePath2, 'r');scriptContent = fread(fid, '*char').';fclose(fid);
        %search for words and change to import the model solved 
        modifiedContent = strrep(scriptContent, 'mmmooo.odb', [new_jjj '.odb']);%read odb solved
        modifiedContent = strrep(modifiedContent, 'abaqusoo.rpt', [new_jjj '_10000.rpt']);%write location file_10000.rpt
        
        %write modified in file 'C:\Users\osama\OneDrive\Desktop\abaqus2\o\' + file_2.py (visulize python file)
        outputFilePath2 = fullfile(outputDirectory, [new_jjj '_2.py']);
        %write modification
        fid = fopen(outputFilePath2, 'w'); fwrite(fid, modifiedContent); fclose(fid);
            % Re-run the second script
            system(['abaqus cae noGUI=' outputFilePath2]);
            
            % Re-read the .rpt file
            rptFilePath = fullfile(outputDirectory, [new_jjj '_10000.rpt']);fid = fopen(rptFilePath, 'r');data = textscan(fid, '%f %f', 'HeaderLines', 2);fclose(fid);
            
            aa = data{2} * -1 / 1000.0;
            forcei = max(aa);
            checki = aa(end) < 0.995 * forcei;
            disp(checki)
        end
        
        % Store the results
        force = [force; forcei];
        check = [check; checki];
    end
end
