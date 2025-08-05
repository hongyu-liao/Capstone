# Docker Deployment Troubleshooting Guide

## Common Errors and Solutions

### 1. `torch-audio` Package Name Error (Fixed)

**Error Message:**
```
ERROR: Could not find a version that satisfies the requirement torch-audio>=2.6.0
```

**Solution:**
- ✅ Fixed `torch-audio` to `torchaudio`
- ✅ Optimized dependency installation order

### 2. Docker Build Failure

**If you encounter build failures, follow these steps:**

#### Quick Fix (Recommended)
Run the fix script:
```cmd
REM Windows users
fix_and_retry.bat

REM Linux/Mac users  
./fix_and_retry.sh
```

#### Manual Fix Steps

1. **Clean Docker cache:**
   ```cmd
   docker system prune -a -f
   docker builder prune -a -f
   ```

2. **Remove old images:**
   ```cmd
   docker image rm pdf-analyzer
   ```

3. **Rebuild:**
   ```cmd
   deploy.bat
   ```

### 3. Network Related Issues

**Symptoms:** Timeout or failure when downloading dependencies

**Solutions:**
- Ensure stable internet connection
- If on corporate network, may need to configure proxy
- Try rebuilding when network conditions are better

### 4. Insufficient Disk Space

**Symptoms:** Out of space errors during build

**Solutions:**
- Ensure at least 15GB free space available
- Clean Docker cache: `docker system prune -a -f`
- Remove unused Docker images: `docker image prune -a -f`

### 5. Insufficient Memory

**Symptoms:** Build process hangs or fails

**Solutions:**
- Close other large applications
- Increase memory allocation in Docker Desktop settings
- Recommend allocating at least 8GB RAM to Docker

## Verify Build Success

After build completion, you should see:
```
✅ Docker image built successfully
```

You can verify the image was created successfully:
```cmd
docker images | findstr pdf-analyzer
```

Should see output similar to:
```
pdf-analyzer    latest    abc123def456    5 minutes ago    8.5GB
```

## Get Build Logs

If you need detailed build logs for troubleshooting:
```cmd
docker build --no-cache -t pdf-analyzer . > build.log 2>&1
```

Then check the `build.log` file for detailed error information.

## Contact Support

If problems persist, please provide:
1. Complete error output
2. Your operating system information
3. Docker Desktop version
4. Available disk space
5. Network environment information (proxy usage)