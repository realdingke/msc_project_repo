-- Generate a sample from a Gaussian distribution
function gaussian (mean, variance)
    return  math.sqrt(-2 * variance * math.log(math.random() + 0.00001)) *
            math.cos(2 * math.pi * math.random()) + mean
end

function createRandomBumpyFloor()
    print ("Generating new random bumpy floor.")
    sim.setThreadAutomaticSwitch(false)

    -- Remove existing bumpy floor if there already is one
    if (heightField ~= nil) then
        sim.setObjectPosition(heightField, heightField, {0.05, 0, 0})
        return
    end
    --  Create random bumpy floor for robot to drive on
    floorSize = 5
    --heightFieldResolution = 0.3
    --heightFieldNoise = 0.00000005
    heightFieldResolution = 0.1
    heightFieldNoise = 0.0000008
    cellsPerSide = floorSize / heightFieldResolution
    cellHeights = {}
    for i=1,cellsPerSide*cellsPerSide,1 do
        table.insert(cellHeights, gaussian(0, heightFieldNoise))
    end
    heightField=sim.createHeightfieldShape(0, 0, cellsPerSide, cellsPerSide, floorSize, cellHeights)
    -- Make the floor invisible
    sim.setObjectInt32Parameter(heightField,10,0)
    sim.setThreadAutomaticSwitch(true)
end






function get_walls()
    -- Disable error reporting
    local savedState=sim.getInt32Parameter(sim.intparam_error_report_mode)
    sim.setInt32Parameter(sim.intparam_error_report_mode,0)
    local N = 1
    while true do
        local handle = sim.getObjectHandle("Wall"..tostring(N))
        if handle <= 0 then
            break
        end

        -- Read position and shape of wall
        -- Assume here that it is thin and oriented either along the x axis or y axis

        -- We can now get the propertries of these walls, e.g....
        local pos = sim.getObjectPosition(handle, -1)
        local res,minx = sim.getObjectFloatParameter(handle,15)
        local res,maxx = sim.getObjectFloatParameter(handle,18)
        local res,miny = sim.getObjectFloatParameter(handle,16)
        local res,maxy = sim.getObjectFloatParameter(handle,19)
    
        --print("Position of Wall " .. tostring(N) .. ": " .. tostring(pos[1]) .. "," .. tostring(pos[2]) .. "," .. tostring(pos[3]))
        --print("minmax", minx, maxx, miny, maxy)
 
        local Ax, Ay, Bx, By
        if (maxx - minx > maxy - miny) then
            print("Wall " ..tostring(N).. " along x axis")
            Ax = pos[1] + minx
            Ay = pos[2]
            Bx = pos[1] + maxx
            By = pos[2]
        else
            print("Wall " ..tostring(N).. " along y axis")
            Ax = pos[1]
            Ay = pos[2] + miny
            Bx = pos[1]
            By = pos[2] + maxy
        end
        print (Ax, Ay, Bx, By)

        walls[N] = {Ax, Ay, Bx, By}
        N = N + 1
    end
    -- enable error reporting
    sim.setInt32Parameter(sim.intparam_error_report_mode,savedState)

    return N - 1
end



-- This function is executed exactly once when the scene is initialised
function sysCall_init()

    tt = sim.getSimulationTime()
    print("Init hello", tt)
    
          
    robotBase=sim.getObjectAssociatedWithScript(sim.handle_self) -- robot handle
    leftMotor=sim.getObjectHandle("leftMotor") -- Handle of the left motor
    rightMotor=sim.getObjectHandle("rightMotor") -- Handle of the right motor
    turretMotor=sim.getObjectHandle("turretMotor") -- Handle of the turret motor
    turretSensor=sim.getObjectHandle("turretSensor")
     
    -- set turret motor speed (radians per second)
    speedtMotor = 2.0
    sim.setJointTargetVelocity(turretMotor, speedtMotor)


    -- Create bumpy floor for robot to drive on
    createRandomBumpyFloor()




    N_WAYPOINTS = 26
    currentWaypoint = 1
    waypoints = {}
    waypoints[1] = {0.5,0}
    waypoints[2] = {1,0}
    waypoints[3] = {1,0.5}
    waypoints[4] = {1,1}
    waypoints[5] = {1,1.5}
    waypoints[6] = {1,2}
    waypoints[7] = {0.5,2}
    waypoints[8] = {0,2}
    waypoints[9] = {-0.5,2}
    waypoints[10] = {-1,2}
    waypoints[11] = {-1,1.5}
    waypoints[12] = {-1,1}
    waypoints[13] = {-1.5,1}
    waypoints[14] = {-2,1}
    waypoints[15] = {-2,0.5}
    waypoints[16] = {-2,0}
    waypoints[17] = {-2,-0.5}
    waypoints[18] = {-1.5,-1}
    waypoints[19] = {-1,-1.5}
    waypoints[20] = {-0.5,-1.5}
    waypoints[21] = {0,-1.5}
    waypoints[22] = {0.5,-1.5}
    waypoints[23] = {1,-1.5} 
    waypoints[24] = {1,-1}
    waypoints[25] = {0.5,-0.5}
    waypoints[26] = {0,0}


    -- Usual rotation rate for wheels (radians per second)
    speedBase = 2.5
    speedBaseL = 0
    speedBaseR = 0
    
    -- Which step are we in?
    -- 0 is a dummy value which is immediately completed
    stepCounter = 0
    stepCompletedFlag = false
    stepCompletedFlagForward = false
    stepList = {}
        

    stepList[1] = {"set_waypoint"}
    stepList[2] = {"turn"}
    stepList[3] = {"stop"}
    stepList[4] = {"forward"}
    stepList[5] = {"stop"}
    stepList[6] = {"repeat"}

    -- Create and initialise arrays for particles, and display them with dummies
    xArray = {}
    yArray = {}
    thetaArray = {}
    sxArray = {}
    syArray = {}
    sthetaArray = {}
    likeliHood = {}
    weightArray = {}
    dummyArray = {}
    N = 100
    for i=1, N do
        xArray[i] = 0
        yArray[i] = 0
        thetaArray[i] = 0
        sxArray[i] = xArray[i]
        syArray[i] = yArray[i]
        sthetaArray[i] = thetaArray[i]
        likeliHood[i] = 1
        weightArray[i] = 1/N
        dummyArray[i] = sim.createDummy(0.05)
        sim.setObjectPosition(dummyArray[i], -1, {0,0,0})
        sim.setObjectOrientation(dummyArray[i], -1, {0,0,0})
    end


    -- Target positions for joints
    motorAngleTargetL = 0.0
    motorAngleTargetR = 0.0
    x = 0.0
    y = 0.0
    theta = 0.0

     -- To calibrate
    motorAnglePerMetre = 24.8
    motorAnglePerRadian = 3.05
    
    -- Standard derivation
    e = 0.0025 -- Distruibtion error in desired forward distance
    f = 0.0001 -- Distruibtion error in robot's heading during forward process
    g = 0.0025 -- Distruibtion error in turning angle during turning
 

    noisyDistance = 0
    -- Data structure for walls
    walls = {}
    -- Fill it by parsing the scene in the GUI
    N_WALLS = get_walls()
    -- walls now is an array of arrays with the {Ax, Ay, Bx, By} wall coordinates
        
    sensorStandardDeviation = 0.1
    sensorVariance = sensorStandardDeviation^2
    noisyDistance = 0
end

function sysCall_sensing()
    
end

function getMaxMotorAngleFromTarget(posL, posR)

    -- How far are the left and right motors from their targets? Find the maximum
    maxAngle = 0
    if (speedBaseL > 0) then
        remaining = motorAngleTargetL - posL
        if (remaining > maxAngle) then
            maxAngle = remaining
        end
    end
    if (speedBaseL < 0) then
        remaining = posL - motorAngleTargetL
        if (remaining > maxAngle) then
            maxAngle = remaining
        end
    end
    if (speedBaseR > 0) then
        remaining = motorAngleTargetR - posR
        if (remaining > maxAngle) then
            maxAngle = remaining
        end
    end
    if (speedBaseR < 0) then
        remaining = posR - motorAngleTargetR
        if (remaining > maxAngle) then
            maxAngle = remaining
        end
    end

    return maxAngle
end


function get_distance(point_1, point_2)
    return math.sqrt((point_1[1] - point_2[1])^2 + (point_1[2] - point_2[2])^2)
end


function calculateLikelihood0(x, y, theta, z)
    sm = 1e309
    m = {}
    k = 0
    
    xhit = 0
    yhit = 0
    for i=1, N_WALLS do
        Ax = walls[i][1]
        Ay = walls[i][2]
        Bx = walls[i][3]
        By = walls[i][4]
        distance = ((By - Ay)*(Ax - x)-(Bx - Ax)*(Ay - y))/((By - Ay)*math.cos(theta)-(Bx - Ax)*math.sin(theta))
        --angle = math.acos(((Ay - By)*math.cos(theta) + (Bx - Ax)*math.sin(theta))/math.sqrt((Ax - Bx)^2 + (Ay - By)^2))
        if (distance>0) then
            wall_index = 0
            xhit = x + distance*math.cos(theta)
            yhit = y + distance*math.sin(theta)
            
            --[[if (xhit >= Ax) and (xhit <= Bx) and (yhit >= Ay) and (yhit <= By) then
                wall_index = i
            elseif (xhit >= Bx) and (xhit <= Ax) and (yhit >= Ay) and (yhit <= By) then 
                wall_index = i
            elseif (xhit >= Bx) and (xhit <= Ax) and (yhit >= By) and (yhit <= Ay) then
                wall_index = i
            elseif (xhit >= Ax) and (xhit <= Bx) and (yhit >= By) and (yhit <= Ay) then
                wall_index = i  
            end]]--
            
            distanceAB = get_distance({Ax, Ay}, {Bx, By})
            distancehitA = get_distance({Ax, Ay}, {xhit, yhit})
            distancehitB = get_distance({Bx, By}, {xhit, yhit})
            if math.abs((distancehitA + distancehitB) - distanceAB) <= 0.001 then
                wall_index = i
            end
            --print(wall_index)
            --k = 0
            if (wall_index ~= 0) then
                k = k + 1
                m[k] = {wall_index, distance}
            end
        end
    end
    for j=1, #m do
        if (m[j][2] < sm) then
            sm = m[j][2]
        end
    end
    --print('xyz     ', x, y, theta)
    --print('xyz     ', xhit, yhit, theta)
    --print('z       ', z)
    --print('in        ', wall_index)
    --print('wall      ', walls[wall_index])
    print('m : \n', m)
    print('length of m         ', #m)
    --print('sm        ',sm)
    --proT = math.exp(-1*(math.rad(90)-beta[wall_index])^2 /(2*sensorVariance)) + 0.0
    proF = math.exp(-1*(z-sm)^2 /(2*sensorVariance)) + 0.01
    
    return proF
end


function calculateLikelihood(x, y, theta, z)
    sm = 1e309
    m = {}
    
    xhit = 0
    yhit = 0
    for i=1, N_WALLS do
        Ax = walls[i][1]
        Ay = walls[i][2]
        Bx = walls[i][3]
        By = walls[i][4]
        distance = ((By - Ay)*(Ax - x)-(Bx - Ax)*(Ay - y))/((By - Ay)*math.cos(theta)-(Bx - Ax)*math.sin(theta))
        --angle = math.acos(((Ay - By)*math.cos(theta) + (Bx - Ax)*math.sin(theta))/math.sqrt((Ax - Bx)^2 + (Ay - By)^2))
        if (distance>0) then
            xhit = x + distance*math.cos(theta)
            yhit = y + distance*math.sin(theta)
            
            --[[if (xhit >= Ax) and (xhit <= Bx) and (yhit >= Ay) and (yhit <= By) then
                wall_index = i
            elseif (xhit >= Bx) and (xhit <= Ax) and (yhit >= Ay) and (yhit <= By) then 
                wall_index = i
            elseif (xhit >= Bx) and (xhit <= Ax) and (yhit >= By) and (yhit <= Ay) then
                wall_index = i
            elseif (xhit >= Ax) and (xhit <= Bx) and (yhit >= By) and (yhit <= Ay) then
                wall_index = i  
            end]]--
            
            distanceAB = get_distance({Ax, Ay}, {Bx, By})
            distancehitA = get_distance({Ax, Ay}, {xhit, yhit})
            distancehitB = get_distance({Bx, By}, {xhit, yhit})
            if math.abs((distancehitA + distancehitB) - distanceAB) <= 0.001 then
                wall_index = i
                table.insert(m, distance)
            end
        end
    end
    for j=1, #m do
        if (m[j] < sm) then
            sm = m[j]
        end
    end
    --print('xyz     ', x, y, theta)
    --print('xyz     ', xhit, yhit, theta)
    --print('z       ', z)
    --print('in        ', wall_index)
    --print('wall      ', walls[wall_index])
    print('m : \n', m)
    print('length of m         ', #m)
    print('sm        ',sm)
    --proT = math.exp(-1*(math.rad(90)-beta[wall_index])^2 /(2*sensorVariance)) + 0.0
    proF = math.exp(-1*(z-sm)^2 /(2*sensorVariance)) + 0.01
    
    return proF
end


function normalising(weights)
    weightSum = 0
    for i=1, #weights do
        weightSum = weightSum + weights[i]
    end
    normalW = {}
    for i=1, #weights do
        normalW[i] = weights[i]/weightSum
    end
    return normalW

end

function sampling(x, y, theta, weights)
    cumW = {}
    sampleX = {}
    sampleY = {}
    sampleT = {}
    cumW[1] = weights[1]
    for i=2, #weights do
        cumW[i] = cumW[i-1] + weights[i]
    end
    for i=1, #cumW do
        rn = math.random()
        if (rn < cumW[1]) then
            sampleX[i] = x[1]
            sampleY[i] = y[1]
            sampleT[i] = theta[1]
        end
        for j=2, #cumW do
            if (cumW[j-1] <= rn) and (rn <cumW[j]) then
                sampleX[i] = x[j]
                sampleY[i] = y[j]
                sampleT[i] = theta[j]
            end
        end
    end 
    return sampleX, sampleY, sampleT
end

function sysCall_actuation() 
    tt = sim.getSimulationTime()
    -- calculate current turret sensor angle in radian
    TurretSensorAngle = (speedtMotor*tt)%math.rad(360)
    --print('Current TurretSensorAngle: ', TurretSensorAngle)
    
        -- Get and plot current angles of motor joints
    posL = sim.getJointPosition(leftMotor)
    posR = sim.getJointPosition(rightMotor)
    
    --result,cleanDistance=sim.readProximitySensor(turretSensor)

    --if (result>0) then
        --noisyDistance= cleanDistance + gaussian(0.0, sensorVariance)
        
        --print ("Depth sensor reading ", noisyDistance)
    --end


    -- Start new step?
    if (stepCompletedFlag == true or stepCompletedFlagForward == true or stepCounter == 0) then
        
        stepCounter = stepCounter + 1
        if (stepCompletedFlag == true) then
            stepCompletedFlag = false
        elseif (stepCompletedFlagForward == true) then
            stepCompletedFlagForward = false
            currentWaypoint = currentWaypoint + 1
        end

        if (currentWaypoint > N_WAYPOINTS) then
            currentWaypoint = 1
        end
        
        newStepType = stepList[stepCounter][1]

        if (newStepType == "repeat") then
            -- Loop back to the first step
            stepCounter = 1
            newStepType = stepList[stepCounter][1]
        end

        print("New step:", stepCounter, newStepType)
        if (newStepType == "set_waypoint") then
            xnewStepAmount = waypoints[currentWaypoint][1] - x
            ynewStepAmount = waypoints[currentWaypoint][2] - y
            print('waypoint:            ',waypoints[currentWaypoint][1],waypoints[currentWaypoint][2] )
            
            stepCompletedFlag = true
        elseif (newStepType == "forward") then
            -- Forward step: set new joint targets
            newStepAmount = math.sqrt(ynewStepAmount^2 + xnewStepAmount^2)
            motorAngleTargetL = posL + newStepAmount * motorAnglePerMetre
            motorAngleTargetR = posR + newStepAmount * motorAnglePerMetre
           
        elseif (newStepType == "turn") then
            -- Turn step: set new targets
            newStepAmount = math.atan2(ynewStepAmount, xnewStepAmount) - theta
            while (newStepAmount < math.rad(-180)) do
                newStepAmount = newStepAmount + 2*math.rad(180)
            end
            while (newStepAmount > math.rad(180)) do
                newStepAmount = newStepAmount - 2*math.rad(180)
            end
            motorAngleTargetL = posL - newStepAmount * motorAnglePerRadian
            motorAngleTargetR = posR + newStepAmount * motorAnglePerRadian
        elseif (newStepType == "stop") then
            print ("Stopping!")
        end
    end


    -- Handle current ongoing step
    stepType = stepList[stepCounter][1]

    if (stepType == "turn") then
        if (newStepAmount >= 0) then
            speedBaseL = -speedBase
            speedBaseR = speedBase
        else
            speedBaseL = speedBase
            speedBaseR = -speedBase
        end
        motorAngleFromTarget = getMaxMotorAngleFromTarget(posL, posR)
        -- Slow down when close
        if (motorAngleFromTarget < 3) then
            speedScaling = 0.2 + 0.8 * motorAngleFromTarget / 3
            speedBaseL = speedBaseL * speedScaling
            speedBaseR = speedBaseR * speedScaling
        end
        if (motorAngleFromTarget == 0) then

            for i=1, N do
                xArray[i] = xArray[i]
                yArray[i] = yArray[i]
                thetaArray[i] = thetaArray[i] + newStepAmount + gaussian(0, math.abs((newStepAmount/math.rad(90))*g))
                --sim.setObjectPosition(dummyArray[i], -1, {xArray[i],yArray[i],0})
                --sim.setObjectOrientation(dummyArray[i], -1, {0,0,thetaArray[i]})
            end

            
            
            stepCompletedFlag = true
            print('turn:     ',newStepAmount)
            print('f:   ', x, y, theta)
        end
    elseif (stepType == "forward") then
        speedBaseL = speedBase
        speedBaseR = speedBase
        motorAngleFromTarget = getMaxMotorAngleFromTarget(posL, posR)
        -- Slow down when close
        if (motorAngleFromTarget < 3) then
            speedScaling = 0.2 + 0.8 * motorAngleFromTarget / 3
            speedBaseL = speedBaseL * speedScaling
            speedBaseR = speedBaseR * speedScaling
        end
        if (motorAngleFromTarget == 0) then
   
            for i=1, N do
                thetaArray[i] = thetaArray[i] + gaussian(0, f)
                xArray[i] = xArray[i]+(gaussian(0, math.abs(newStepAmount*e)) + newStepAmount)*math.cos(thetaArray[i])
                yArray[i] = yArray[i]+(gaussian(0, math.abs(newStepAmount*e)) + newStepAmount)*math.sin(thetaArray[i])
                
            end
            --result,cleanDistance=sim.readProximitySensor(turretSensor)

            --if (result>0) then
                --noisyDistance = cleanDistance + gaussian(0.0, sensorVariance)
                --for i=1, N do 
                    --likeliHood[i] = calculateLikelihood(xArray[i], yArray[i], thetaArray[i], noisyDistance)
                    --weightArray[i] = likeliHood[i]*weightArray[i]
                --end
        --print ("Depth sensor reading ", noisyDistance)
            --end

            --weightArray = normalising(weightArray)
            --sxArray, syArray, sthetaArray = sampling(xArray, yArray, thetaArray, weightArray)
            --for i=1, N do
                --xArray[i] = sxArray[i]
                --yArray[i] = syArray[i]
                --thetaArray[i] = sthetaArray[i]
            --end
            
            

            stepCompletedFlagForward = true
            print('forward     ',newStepAmount)
            print('g:   ', x, y, theta)
        end
    elseif (stepType == "stop") then
        speedBaseL = 0
        speedBaseR = 0
        
        tt = sim.getSimulationTime()
        -- calculate current turret sensor angle in radian
        TurretSensorAngle = (speedtMotor*tt)%math.rad(360)

        -- Check to see if the robot is stationary to within a small threshold
        linearVelocity,angularVelocity=sim.getVelocity(robotBase)
        vLin = math.sqrt(linearVelocity[1]^2 + linearVelocity[2]^2 + linearVelocity[3]^2)
        vAng = math.sqrt(angularVelocity[1]^2 + angularVelocity[2]^2 + angularVelocity[3]^2)
        --print ("stop", linearVelocity, vLin, vAng)
    
        if (vLin < 0.001 and vAng < 0.01) then
            result,cleanDistance=sim.readProximitySensor(turretSensor)

            if (result>0) then
                noisyDistance = cleanDistance + gaussian(0.0, sensorVariance)
                for i=1, N do 
                    likeliHood[i] = calculateLikelihood(xArray[i], yArray[i], thetaArray[i]+TurretSensorAngle, noisyDistance)
                    weightArray[i] = likeliHood[i]*weightArray[i]
                end
        --print ("Depth sensor reading ", noisyDistance)
            end

            weightArray = normalising(weightArray)
            sxArray, syArray, sthetaArray = sampling(xArray, yArray, thetaArray, weightArray)
            for i=1, N do
                xArray[i] = sxArray[i]
                yArray[i] = syArray[i]
                thetaArray[i] = sthetaArray[i]
                sim.setObjectPosition(dummyArray[i], -1, {xArray[i],yArray[i],0})
                sim.setObjectOrientation(dummyArray[i], -1, {0,0,thetaArray[i]})
            end
            xMean = 0
            yMean = 0
            tMean = 0
            for i=1, N do
                weightArray[i] = 1/N
                xMean = xMean + xArray[i]*weightArray[i]
                yMean = yMean + yArray[i]*weightArray[i]
                tMean = tMean + thetaArray[i]*weightArray[i]
            end
            x = xMean
            y = yMean
            theta = tMean
            stepCompletedFlag = true
        end
    end

    -- Set the motor velocities for the current step
    sim.setJointTargetVelocity(leftMotor,speedBaseL)
    sim.setJointTargetVelocity(rightMotor,speedBaseR)
    
    result,cleanDistance=sim.readProximitySensor(turretSensor)
    
    --linearPosition = sim.getObjectPosition(robotBase,-1)
    --print(linearPosition)
    --print(x,y,theta)

end

function sysCall_cleanup()
    --simUI.destroy(ui)
end 
