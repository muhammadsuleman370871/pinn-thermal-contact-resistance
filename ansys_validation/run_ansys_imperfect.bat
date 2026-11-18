@echo off
echo ============================================
echo Running ANSYS MAPDL - IMPERFECT CONTACT
echo Input: validate_tcr_imperfect.inp
echo Output: ansys_imperfect_results.txt
echo ============================================

cd /d "C:\Users\muham\Desktop"

REM Try ANSYS 2024 R2 (v242) - known working version
if exist "C:\Program Files\ANSYS Inc\v242\ansys\bin\winx64\MAPDL.exe" (
    echo Found ANSYS 2024 R2
    "C:\Program Files\ANSYS Inc\v242\ansys\bin\winx64\MAPDL.exe" -b -np 1 -dir "C:\Users\muham\Desktop" -j "tcr_imperfect" -i "C:\Users\muham\Desktop\validate_tcr_imperfect.inp" -o "C:\Users\muham\Desktop\ansys_imperfect_log.out"
    goto done
)

REM Try ANSYS 2024 R1 (v241)
if exist "C:\Program Files\ANSYS Inc\v241\ansys\bin\winx64\MAPDL.exe" (
    echo Found ANSYS 2024 R1
    "C:\Program Files\ANSYS Inc\v241\ansys\bin\winx64\MAPDL.exe" -b -np 1 -dir "C:\Users\muham\Desktop" -j "tcr_imperfect" -i "C:\Users\muham\Desktop\validate_tcr_imperfect.inp" -o "C:\Users\muham\Desktop\ansys_imperfect_log.out"
    goto done
)

echo ERROR: Could not find ANSYS MAPDL installation!
goto end

:done
echo.
echo ============================================
echo ANSYS imperfect-contact run complete!
echo Check ansys_imperfect_results.txt on Desktop
echo ============================================

:end
