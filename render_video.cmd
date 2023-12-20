REM @ECHO OFF

IF "%1"=="" (
  ECHO "Usage: %~nx0 path\to\folder"
  ECHO "  e.g. %~nx0 log\default"
  EXIT /B
)

SET FFMPEG_BIN=ffmpeg.exe
SET FPS=30
SET OUT_FILE=%1\seq.mp4

%FFMPEG_BIN% -y -framerate %FPS% -i %1\%%8d.png -crf 20 -c:v libx264 -pix_fmt yuv420p %OUT_FILE%

ECHO.
ECHO ^>^> Save video to %OUT_FILE%
ECHO ^>^> Done~
