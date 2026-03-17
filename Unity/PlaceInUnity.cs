using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
// using Newtonsoft.Json; // Using Unity's built-in JsonUtility instead
using System.Linq;

[System.Serializable]
public class ObjectSize
{
    public float length;
    public float width;
    public float height;
}

[System.Serializable]
public class ObjectPosition
{
    public float x;
    public float y;
    public float z;
}

[System.Serializable]
public class ObjectRotation
{
    public float z_angle;
}

[System.Serializable]
public class RoomObject
{
    public string new_object_id;
    public string style;
    public string material;
    public ObjectSize size_in_meters;
    public bool is_on_the_floor;
    public string facing;
    public ObjectPosition position;
    public ObjectRotation rotation;
}

[System.Serializable]
public class SceneGraphData
{
    public List<RoomObject> objects_in_room;
}

public class PlaceInUnity : MonoBehaviour
{
    [Header("Scene Configuration")]
    public string sceneGraphPath = "Assets/Resources/scene_graph.json";
    public string equipmentFolder = "Assets/Resources/Equipment";     // Equipment prefabs in Unity Resources
    public string robotsFolder = "Assets/Resources/Robots";           // Robot prefabs in Unity Resources
    
    [Header("Materials")]
    public Material floorMaterial;
    public Material wallMaterial;
    public Color floorColor = new Color(0.28f, 0.42f, 0.58f);
    
    [Header("Workspace Structure")]
    public bool createWalls = false;
    public bool createCeiling = false;
    
    [Header("Robot Settings")]
    public bool disableArticulationBodiesOnLoad = true;
    
    private Dictionary<string, RoomObject> objectsInRoom = new Dictionary<string, RoomObject>();
    private Dictionary<string, RoomObject> roomElements = new Dictionary<string, RoomObject>();
    private Dictionary<string, string> assetPaths = new Dictionary<string, string>();
    private HashSet<string> robotObjectIds = new HashSet<string>();
    private GameObject floorObject;
    private Transform floorAnchor; // Unity object to hold floor items without inheriting plane scale
    
    void Start()
    {
        LogExpectedStructure();
        LoadSceneGraph();
        FindEquipmentFiles();
        FindRobotFiles();
        CreateWorkspace();
        PlaceObjects();
    }
    
    void LogExpectedStructure()
    {
        Debug.Log("Expected Unity Project Structure:");
        Debug.Log("Assets/Resources/Equipment/ - Equipment prefabs (e.g., 55Gal_Drum_FY001.prefab)");
        Debug.Log("Assets/Resources/Robots/ - Robot prefabs (e.g., ABB/abb_crb15000_support/abb_crb15000_5_95.prefab)");
        Debug.Log("Assets/Resources/ - Scene graph JSON files from Python system");
    }
    
    void LoadSceneGraph()
    {
        string fullPath = Path.Combine(Application.dataPath, sceneGraphPath.Replace("Assets/", ""));
        
        if (!File.Exists(fullPath))
        {
            Debug.LogError($"Scene graph file not found: {fullPath}");
            return;
        }
        
        try
        {
            string jsonContent = File.ReadAllText(fullPath);
            
            // Wrap the array in an object for JsonUtility compatibility
            if (jsonContent.TrimStart().StartsWith("["))
            {
                jsonContent = "{\"objects_in_room\":" + jsonContent + "}";
            }
            
            SceneGraphData data = JsonUtility.FromJson<SceneGraphData>(jsonContent);
            
            foreach (var item in data.objects_in_room)
            {
                if (IsWallOrRoomElement(item.new_object_id))
                {
                    roomElements[item.new_object_id] = item;
                }
                else
                {
                    objectsInRoom[item.new_object_id] = item;
                }
            }
            
            Debug.Log($"Loaded {objectsInRoom.Count} objects and {roomElements.Count} room elements from scene graph");
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Error loading scene graph: {e.Message}");
        }
    }
    
    bool IsWallOrRoomElement(string objectId)
    {
        string[] roomElements = { "south_wall", "north_wall", "east_wall", "west_wall", "middle of the room", "ceiling" };
        return roomElements.Contains(objectId);
    }

    bool IsRobotIdentifier(string objectId)
    {
        if (string.IsNullOrEmpty(objectId))
            return false;

        string lower = objectId.ToLowerInvariant();

        if (lower.Contains("abb_") || lower.Contains("kuka_") || lower.Contains("fanuc_") ||
            lower.Contains("yaskawa_") || lower.Contains("motoman_") || lower.Contains("staubli_") ||
            lower.Contains("comau_") || lower.Contains("kawasaki_") || lower.Contains("hanwha_") ||
            lower.Contains("doosan_"))
        {
            return true;
        }

        if (lower.StartsWith("ur") && lower.Length > 2 && char.IsDigit(lower[2]))
            return true;

        if (lower.StartsWith("robot_") || lower.Contains("_robot_"))
            return true;

        return false;
    }
    
    string FindEquipmentKey(string objectId)
    {
        if (string.IsNullOrEmpty(objectId))
            return null;
        
        string baseObjectId = RemoveTrailingIndex(objectId);
        
        // Try exact match first
        if (assetPaths.ContainsKey(objectId))
            return objectId;
        if (assetPaths.ContainsKey(baseObjectId))
            return baseObjectId;
            
        // Try case-insensitive match
        foreach (var key in assetPaths.Keys)
        {
            if (string.Equals(key, objectId, System.StringComparison.OrdinalIgnoreCase) ||
                string.Equals(key, baseObjectId, System.StringComparison.OrdinalIgnoreCase))
                return key;
        }
        
        // Try matching with underscores converted to capitalize
        string capitalizedId = CapitalizeEquipmentName(baseObjectId);
        if (assetPaths.ContainsKey(capitalizedId))
            return capitalizedId;
            
        string simplifiedObjectId = SimplifyName(baseObjectId);
        if (!string.IsNullOrEmpty(simplifiedObjectId))
        {
            foreach (var key in assetPaths.Keys)
            {
                if (SimplifyName(key) == simplifiedObjectId)
                    return key;
            }
        }
        
        // Try fuzzy matching for common patterns
        foreach (var key in assetPaths.Keys)
        {
            if (FuzzyMatchEquipment(key, baseObjectId))
                return key;
        }
        
        return null;
    }
    
    string CapitalizeEquipmentName(string input)
    {
        if (string.IsNullOrEmpty(input))
            return input;
            
        // Split by underscore, capitalize each part, rejoin
        string[] parts = input.Split('_');
        for (int i = 0; i < parts.Length; i++)
        {
            if (parts[i].Length > 0)
            {
                parts[i] = char.ToUpper(parts[i][0]) + parts[i].Substring(1).ToLower();
            }
        }
        return string.Join("_", parts);
    }
    
    string RemoveTrailingIndex(string input)
    {
        if (string.IsNullOrEmpty(input))
            return input;
        
        return System.Text.RegularExpressions.Regex.Replace(input, "_\\d+$", "");
    }
    
    string SimplifyName(string input)
    {
        if (string.IsNullOrEmpty(input))
            return input;
        
        var simplifiedChars = input.ToLower().Where(char.IsLetterOrDigit);
        return new string(simplifiedChars.ToArray());
    }
    
    string NormalizeMatchString(string input)
    {
        if (string.IsNullOrEmpty(input))
            return input;
        
        string lowered = input.ToLower();
        lowered = System.Text.RegularExpressions.Regex.Replace(lowered, "_fy\\d+", "");
        lowered = lowered.Replace("_support", "").Replace("_robot", "");
        lowered = RemoveTrailingIndex(lowered);
        
        return lowered;
    }
    
    string[] GetSearchTokens(string input)
    {
        if (string.IsNullOrEmpty(input))
            return new string[0];
        
        var tokens = input
            .Split(new char[] { '_', '-', ' ' }, System.StringSplitOptions.RemoveEmptyEntries)
            .Select(t => t.ToLower())
            .Where(t => t.Length >= 3)
            .ToArray();
        
        if (tokens.Length == 0)
        {
            string simplified = SimplifyName(input);
            if (!string.IsNullOrEmpty(simplified))
                return new string[] { simplified };
        }
        
        return tokens;
    }
    
    bool FuzzyMatchEquipment(string equipmentName, string objectId)
    {
        string cleanEquipment = NormalizeMatchString(equipmentName);
        string cleanObject = NormalizeMatchString(objectId);
            
        // For robots, try matching without manufacturer prefix
        if (cleanEquipment.StartsWith("abb_") || cleanEquipment.StartsWith("kuka_") || 
            cleanEquipment.StartsWith("ur") || cleanEquipment.StartsWith("robotiq_"))
        {
            // Try matching just the model part
            string[] equipmentParts = cleanEquipment.Split('_');
            string[] objectParts = cleanObject.Split('_');
            
            // Check if any significant part matches
            foreach (string equipPart in equipmentParts)
            {
                foreach (string objPart in objectParts)
                {
                    if (equipPart.Length > 2 && objPart.Length > 2 && 
                        (equipPart.Contains(objPart) || objPart.Contains(equipPart)))
                    {
                        return true;
                    }
                }
            }
        }

        // Require that all object tokens match equipment tokens to avoid overly broad matches
        if (cleanEquipment == cleanObject && cleanObject.Length > 0)
        {
            return true;
        }

        string[] equipmentTokens = GetSearchTokens(cleanEquipment);
        string[] objectTokens = GetSearchTokens(cleanObject);

        if (objectTokens.Length == 0 || equipmentTokens.Length == 0)
        {
            return false;
        }

        foreach (string objToken in objectTokens)
        {
            bool tokenMatched = false;
            foreach (string equipToken in equipmentTokens)
            {
                if (objToken.Length > 0 && equipToken.Length > 0 &&
                    (equipToken.Contains(objToken) || objToken.Contains(equipToken)))
                {
                    tokenMatched = true;
                    break;
                }
            }

            if (!tokenMatched)
            {
                return false;
            }
        }

        return true;
    }
    
    void FindEquipmentFiles()
    {
        string equipmentPath = Path.Combine(Application.dataPath, equipmentFolder.Replace("Assets/", ""));
        
        if (!Directory.Exists(equipmentPath))
        {
            Debug.LogError($"Equipment folder not found: {equipmentPath}");
            return;
        }
        
        // Find prefab files in the equipment directory
        string[] prefabFiles = Directory.GetFiles(equipmentPath, "*.prefab", SearchOption.TopDirectoryOnly);
        
        foreach (string file in prefabFiles)
        {
            string fileName = Path.GetFileNameWithoutExtension(file);
            // Remove Unity prefab suffix patterns like _FY001
            string cleanName = System.Text.RegularExpressions.Regex.Replace(fileName, "_FY\\d+$", "");
            
            // Store as Resources path for runtime loading
            string resourcePath = "Equipment/" + fileName;
            
            if (!assetPaths.ContainsKey(cleanName))
            {
                assetPaths[cleanName] = resourcePath;
                Debug.Log($"Added equipment: {cleanName} -> {resourcePath}");
            }
        }
        
        Debug.Log($"Found {prefabFiles.Length} equipment prefabs");
    }
    
    void FindRobotFiles()
    {
        string robotsPath = Path.Combine(Application.dataPath, robotsFolder.Replace("Assets/", ""));
        
        if (!Directory.Exists(robotsPath))
        {
            Debug.LogError($"Robots folder not found: {robotsPath}");
            return;
        }
        
        // Find prefab files in the robots directory, excluding URDF component parts
        string[] prefabFiles = Directory.GetFiles(robotsPath, "*.prefab", SearchOption.AllDirectories);
        
        int robotCount = 0;
        foreach (string file in prefabFiles)
        {
            // Skip URDF component files (collision meshes, links, etc.)
            if (file.Contains("/urdf/") || file.Contains("\\urdf\\") || 
                file.Contains("/collision/") || file.Contains("\\collision\\") || 
                file.Contains("/visual/") || file.Contains("\\visual\\"))
                continue;
                
            string fileName = Path.GetFileNameWithoutExtension(file);
            
            // Clean robot name - remove variant suffixes but keep manufacturer prefixes
            string cleanName = fileName;
            if (cleanName.Contains("_trans Variant"))
                cleanName = cleanName.Replace("_trans Variant", "");
            
            // Convert to Resources path format
            string resourcePath = Path.GetRelativePath(Path.Combine(Application.dataPath, "Resources"), file).Replace("\\", "/");
            resourcePath = resourcePath.Replace(".prefab", ""); // Remove extension for Resources.Load
            
            if (!assetPaths.ContainsKey(cleanName))
            {
                assetPaths[cleanName] = resourcePath;
                Debug.Log($"Added robot: {cleanName} -> {resourcePath}");
                robotCount++;
            }
        }
        
        Debug.Log($"Found {robotCount} robot prefabs");
    }
    
    void CreateWorkspace()
    {
        // Create floor
        CreateFloor();
        
        // Temporarily commented out for better visibility in Unity
        if (createWalls)
        {
            CreateWalls();
        }
        
        if (createCeiling)
        {
            CreateCeiling();
        }
    }
    
    void CreateFloor()
    {
        // Get floor dimensions from walls
        if (!roomElements.ContainsKey("south_wall") || !roomElements.ContainsKey("east_wall"))
        {
            Debug.LogError("Cannot create floor: wall elements not found!");
            return;
        }
        
        var southWall = roomElements["south_wall"];
        var eastWall = roomElements["east_wall"];
        
        float roomWidth = southWall.size_in_meters.length;  // South wall length = room width
        float roomDepth = eastWall.size_in_meters.length;   // East wall length = room depth
        
        floorObject = GameObject.CreatePrimitive(PrimitiveType.Plane);
        floorObject.name = "Floor";
        floorObject.transform.position = new Vector3(0, 0, 0); // Center the floor at Unity origin
        floorObject.transform.localScale = new Vector3(roomWidth / 10f, 1, roomDepth / 10f);
        
        var floorRenderer = floorObject.GetComponent<Renderer>();
        Material runtimeMaterial = null;

        if (floorMaterial != null)
        {
            runtimeMaterial = new Material(floorMaterial);
        }
        else if (floorRenderer.sharedMaterial != null)
        {
            runtimeMaterial = new Material(floorRenderer.sharedMaterial);
        }
        else
        {
            Shader defaultShader = Shader.Find("Standard");
            if (defaultShader == null)
            {
                defaultShader = Shader.Find("Universal Render Pipeline/Lit");
            }

            if (defaultShader != null)
            {
                runtimeMaterial = new Material(defaultShader);
            }
        }

        if (runtimeMaterial != null)
        {
            runtimeMaterial.color = floorColor;
            floorRenderer.material = runtimeMaterial;
        }
        
        // Create or update neutral anchor that cancels the plane's non-uniform scale
        if (floorAnchor == null)
        {
            floorAnchor = new GameObject("FloorAnchor").transform;
        }
        floorAnchor.SetParent(transform, false);
        floorAnchor.position = floorObject.transform.position;
        floorAnchor.rotation = floorObject.transform.rotation;
        floorAnchor.localScale = Vector3.one;
        
        Debug.Log($"Created floor: {roomWidth}x{roomDepth}m");
    }
    
    void CreateWalls()
    {
        string[] wallIds = {"south_wall", "north_wall", "east_wall", "west_wall"};
        
        foreach (string wallId in wallIds)
        {
            if (roomElements.ContainsKey(wallId))
            {
                var wall = roomElements[wallId];
                CreateWallFromData(wallId, wall);
            }
            else
            {
                Debug.LogWarning($"Wall {wallId} not found in room elements!");
            }
        }
    }
    
    void CreateWallFromData(string wallId, RoomObject wallData)
    {
        GameObject wall = GameObject.CreatePrimitive(PrimitiveType.Plane);
        wall.name = wallId;
        
        // Set position from JSON data with coordinate system conversion
        // Convert from JSON coordinate system to Unity coordinate system (centered at origin)
        var southWall = roomElements["south_wall"];
        var eastWall = roomElements["east_wall"];
        float roomWidth = southWall.size_in_meters.length;
        float roomDepth = eastWall.size_in_meters.length;
        
        wall.transform.position = new Vector3(
            wallData.position.x - roomWidth / 2f,   // JSON X → Unity X (west/east)
            wallData.position.z,                    // JSON Z → Unity Y (height)
            wallData.position.y - roomDepth / 2f    // JSON Y → Unity Z (south/north)
        );
        
        // Set scale based on wall dimensions
        float width = wallData.size_in_meters.length;
        float height = wallData.size_in_meters.height;
        wall.transform.localScale = new Vector3(width / 10f, height / 10f, 1);
        
        // Set rotation from JSON data
        wall.transform.rotation = Quaternion.Euler(90, wallData.rotation.z_angle, 0);
        
        if (wallMaterial != null)
        {
            wall.GetComponent<Renderer>().material = wallMaterial;
        }
        
        Debug.Log($"Created {wallId}: {width}x{height}m at position {wallData.position.x},{wallData.position.y},{wallData.position.z}");
    }
    
    
    void CreateCeiling()
    {
        if (roomElements.ContainsKey("ceiling"))
        {
            var ceilingData = roomElements["ceiling"];
            GameObject ceiling = GameObject.CreatePrimitive(PrimitiveType.Plane);
            ceiling.name = "ceiling";
            
            // Set position from JSON data with coordinate system conversion
            var southWall = roomElements["south_wall"];
            var eastWall = roomElements["east_wall"];
            float roomWidth = southWall.size_in_meters.length;
            float roomDepth = eastWall.size_in_meters.length;
            
            ceiling.transform.position = new Vector3(
                ceilingData.position.x - roomWidth / 2f,   // JSON X → Unity X (west/east)
                ceilingData.position.z,                    // JSON Z → Unity Y (height)
                ceilingData.position.y - roomDepth / 2f    // JSON Y → Unity Z (south/north)
            );
            
            // Set scale based on ceiling dimensions
            float width = ceilingData.size_in_meters.length;
            float depth = ceilingData.size_in_meters.width;
            ceiling.transform.localScale = new Vector3(width / 10f, 1, depth / 10f);
            ceiling.transform.rotation = Quaternion.Euler(180, 0, 0);
            
            if (wallMaterial != null)
            {
                ceiling.GetComponent<Renderer>().material = wallMaterial;
            }
            
            Debug.Log($"Created ceiling: {width}x{depth}m at position {ceilingData.position.x},{ceilingData.position.y},{ceilingData.position.z}");
        }
        else
        {
            Debug.LogWarning("Ceiling element not found in room elements!");
        }
    }
    
    void PlaceObjects()
    {
        foreach (var kvp in objectsInRoom)
        {
            string objectId = kvp.Key;
            RoomObject roomObj = kvp.Value;
            
            // Find the corresponding equipment prefab
            string matchedKey = FindEquipmentKey(objectId);
            if (!string.IsNullOrEmpty(matchedKey) && assetPaths.ContainsKey(matchedKey))
            {
                LoadAndPlaceEquipment(objectId, roomObj, assetPaths[matchedKey]);
            }
            else
            {
                Debug.LogWarning($"Equipment prefab not found for: {objectId}");
                Debug.LogWarning($"Available equipment: {string.Join(", ", assetPaths.Keys)}");
                CreatePlaceholderObject(objectId, roomObj);
            }
        }
    }
    
    void LoadAndPlaceEquipment(string objectId, RoomObject roomObj, string resourcePath)
    {
        GameObject prefab = null;
        
        // Try loading from Resources first (works in both editor and build)
        prefab = Resources.Load<GameObject>(resourcePath);
        
        #if UNITY_EDITOR
        // In editor, fallback to AssetDatabase if Resources.Load fails
        if (prefab == null)
        {
            // Try constructing full asset path
            string fullAssetPath = "";
            if (resourcePath.StartsWith("Equipment/"))
            {
                fullAssetPath = equipmentFolder + "/" + resourcePath.Substring(10) + ".prefab";
            }
            else if (resourcePath.StartsWith("Robots/"))
            {
                fullAssetPath = robotsFolder + "/" + resourcePath.Substring(7) + ".prefab";
            }
            
            if (!string.IsNullOrEmpty(fullAssetPath))
            {
                prefab = UnityEditor.AssetDatabase.LoadAssetAtPath<GameObject>(fullAssetPath);
            }
        }
        #endif
        
        bool isRobotResource = resourcePath.StartsWith("Robots/");
        if (isRobotResource)
        {
            robotObjectIds.Add(objectId);
        }

        if (prefab != null)
        {
            GameObject instance = Instantiate(prefab);
            if (disableArticulationBodiesOnLoad)
            {
                DisableArticulationBodies(instance);
            }
            ConfigureObject(instance, objectId, roomObj);
            Debug.Log($"Successfully loaded and placed: {objectId} from Resources path: {resourcePath}");
        }
        else
        {
            Debug.LogWarning($"Could not load prefab from Resources: {resourcePath} for object: {objectId}");
            CreatePlaceholderObject(objectId, roomObj);
        }
    }
    
    void CreatePlaceholderObject(string objectId, RoomObject roomObj)
    {
        GameObject placeholder = GameObject.CreatePrimitive(PrimitiveType.Cube);
        placeholder.name = objectId + "_placeholder";
        
        // Color code by equipment type
        Renderer renderer = placeholder.GetComponent<Renderer>();
        string lowerObjectId = objectId.ToLower();
        bool isRobot = IsRobotIdentifier(objectId);
        if (isRobot)
        {
            robotObjectIds.Add(objectId);
        }
        
        // Robot color coding
        if (lowerObjectId.Contains("abb_")) renderer.material.color = Color.red;
        else if (lowerObjectId.Contains("kuka_")) renderer.material.color = new Color(1f, 0.5f, 0f); // orange
        else if (lowerObjectId.Contains("ur") && (lowerObjectId.Contains("5") || lowerObjectId.Contains("10"))) renderer.material.color = Color.blue;
        else if (lowerObjectId.Contains("robotiq")) renderer.material.color = Color.cyan;
        // Equipment color coding
        else if (lowerObjectId.Contains("conveyor")) renderer.material.color = Color.green;
        else if (lowerObjectId.Contains("table")) renderer.material.color = new Color(0.6f, 0.4f, 0.2f); // brown
        else if (lowerObjectId.Contains("pallet")) renderer.material.color = Color.yellow;
        else if (lowerObjectId.Contains("shelving")) renderer.material.color = Color.blue;
        else if (lowerObjectId.Contains("safety")) renderer.material.color = Color.red;
        else renderer.material.color = Color.gray;
        
        ConfigureObject(placeholder, objectId, roomObj);
        
        Debug.LogWarning($"Created placeholder for: {objectId} (could not find matching prefab)");
    }
    
    void DisableArticulationBodies(GameObject obj)
    {
        ArticulationBody[] articulations = obj.GetComponentsInChildren<ArticulationBody>(true);
        if (articulations.Length == 0)
            return;
        
        foreach (var body in articulations)
        {
            body.enabled = false;
        }
        
        Debug.Log($"Disabled {articulations.Length} articulation bodies for {obj.name}");
    }
    
    void ConfigureObject(GameObject obj, string objectId, RoomObject roomObj)
    {
        obj.name = objectId;
        
        // Set scale first to get accurate bounds
        if (roomObj.size_in_meters != null)
        {
            RescaleObject(obj, roomObj.size_in_meters);
        }

        bool isRobotObject = robotObjectIds.Contains(objectId) || IsRobotIdentifier(objectId);

        float baseYaw = 0f;
        bool usedRotationOverride = false;

        if (isRobotObject && roomObj.rotation != null)
        {
            baseYaw = roomObj.rotation.z_angle;
            usedRotationOverride = true;
        }
        else if (roomObj.size_in_meters != null)
        {
            baseYaw = DetermineSizeAlignmentYaw(obj, roomObj.size_in_meters);
        }

        float finalYaw = baseYaw;

        if (isRobotObject)
        {
            if (!usedRotationOverride && TryGetFacingYaw(roomObj.facing, out float facingYaw))
            {
                finalYaw += facingYaw;
            }
        }
        else
        {
            if (TryGetFacingYaw(roomObj.facing, out float facingYaw))
            {
                finalYaw += facingYaw;
            }
            else if (roomObj.rotation != null)
            {
                finalYaw += roomObj.rotation.z_angle;
            }
        }

        finalYaw = NormalizeAngle(finalYaw);
        obj.transform.rotation = Quaternion.Euler(0f, finalYaw, 0f);
        Debug.Log($"Orientation applied to {objectId}: baseYaw={baseYaw:F2}, finalYaw={finalYaw:F2} (isRobot={isRobotObject}, rotationOverride={usedRotationOverride}, facing={roomObj.facing})");

        // Set position with coordinate system conversion
        if (roomObj.position != null)
        {
            // Convert from JSON coordinate system (origin at room top-left) to Unity coordinate system (origin at room center)
            // JSON: (0,0) = top-left corner, Y axis down
            // Unity: (0,0) = room center, Y axis up
            var southWall = roomElements["south_wall"];
            var eastWall = roomElements["east_wall"];
            float roomWidth = southWall.size_in_meters.length;
            float roomDepth = eastWall.size_in_meters.length;
            
            Vector3 targetPosition = new Vector3(
                roomObj.position.x - roomWidth / 2f,   // JSON X → Unity X (west/east)
                roomObj.position.z,                    // JSON Z → Unity Y (height)
                roomObj.position.y - roomDepth / 2f    // JSON Y → Unity Z (south/north)
            );
            
            // If object is on the floor, anchor it to the floor plane but keep world coordinates
            if (roomObj.is_on_the_floor && floorAnchor != null)
            {
                // Use provided height and then align object's bottom with floor plane
                targetPosition.y = roomObj.position != null ? roomObj.position.z : floorAnchor.position.y;
                obj.transform.position = targetPosition;
                obj.transform.SetParent(floorAnchor, true); // Preserve world transform without inheriting floor scale
                
                Bounds objBounds = GetObjectBounds(obj);
                float floorHeight = floorAnchor.position.y;
                float bottomOffset = floorHeight - objBounds.min.y;
                if (Mathf.Abs(bottomOffset) > 0.001f)
                {
                    obj.transform.position += new Vector3(0f, bottomOffset, 0f);
                }
            }
            else
            {
                obj.transform.position = targetPosition;
                obj.transform.SetParent(null);
            }
        }
        
        Debug.Log($"Placed object: {objectId} at localPos {obj.transform.localPosition}, worldPos {obj.transform.position} (on_floor: {roomObj.is_on_the_floor})");
        Debug.Log($"JSON position: ({roomObj.position.x}, {roomObj.position.y}, {roomObj.position.z})");
    }
    
    void RescaleObject(GameObject obj, ObjectSize targetSize)
    {
        // Get the current bounds
        Quaternion originalRotation = obj.transform.rotation;
        obj.transform.rotation = Quaternion.identity;
        
        Bounds initialBounds = GetObjectBounds(obj);
        
        if (initialBounds.size.magnitude <= Mathf.Epsilon)
        {
            obj.transform.rotation = originalRotation;
            Debug.LogWarning($"Cannot rescale {obj.name}: bounds magnitude is zero.");
            return;
        }
        
        Vector3 originalScale = obj.transform.localScale;
        Vector3 originalDimensions = initialBounds.size;
        
        if (originalDimensions.sqrMagnitude <= Mathf.Epsilon)
        {
            obj.transform.rotation = originalRotation;
            Debug.LogWarning($"Cannot rescale {obj.name}: original bounds too small.");
            return;
        }
        
        Vector3[] targetPermutations = new Vector3[]
        {
            new Vector3(targetSize.length, targetSize.height, targetSize.width),
            new Vector3(targetSize.width, targetSize.height, targetSize.length)
        };
        
        float bestError = float.MaxValue;
        Vector3 bestScale = originalScale;
        Vector3 bestTarget = targetPermutations[0];
        float bestFactor = 1f;
        
        foreach (var targetDimensions in targetPermutations)
        {
            float denominator = Vector3.Dot(originalDimensions, originalDimensions);
            if (denominator <= Mathf.Epsilon)
                continue;
            
            float numerator = Vector3.Dot(originalDimensions, targetDimensions);
            float scaleFactor = numerator / denominator;
            scaleFactor = Mathf.Max(scaleFactor, 1e-4f);
            
            Vector3 candidateScale = originalScale * scaleFactor;
            obj.transform.localScale = candidateScale;
            Bounds candidateBounds = GetObjectBounds(obj);
            
            float error = Mathf.Abs(candidateBounds.size.x - targetDimensions.x) +
                          Mathf.Abs(candidateBounds.size.y - targetDimensions.y) +
                          Mathf.Abs(candidateBounds.size.z - targetDimensions.z);
            
            Debug.Log($"Uniform rescale trial {obj.name}: target=({targetDimensions.x:F3},{targetDimensions.y:F3},{targetDimensions.z:F3}), result=({candidateBounds.size.x:F3},{candidateBounds.size.y:F3},{candidateBounds.size.z:F3}), factor={scaleFactor:F4}, error={error:F4}");
            
            if (error < bestError)
            {
                bestError = error;
                bestScale = candidateScale;
                bestTarget = targetDimensions;
                bestFactor = scaleFactor;
            }
        }
        
        obj.transform.localScale = bestScale;
        Bounds finalBounds = GetObjectBounds(obj);
        obj.transform.rotation = originalRotation;
        
        Debug.Log($"Rescaled {obj.name} uniformly: target≈({bestTarget.x:F3},{bestTarget.y:F3},{bestTarget.z:F3}), actual=({finalBounds.size.x:F3},{finalBounds.size.y:F3},{finalBounds.size.z:F3}), factor={bestFactor:F4}");
    }
    
    float DetermineSizeAlignmentYaw(GameObject obj, ObjectSize targetSize)
    {
        Quaternion originalRotation = obj.transform.rotation;
        float[] yawCandidates = { 0f, 90f, 180f, 270f };
        float bestYaw = 0f;
        float bestError = float.MaxValue;
        
        foreach (float yaw in yawCandidates)
        {
            obj.transform.rotation = Quaternion.Euler(0f, yaw, 0f);
            Bounds bounds = GetObjectBounds(obj);
            
            Vector2 actual = new Vector2(bounds.size.x, bounds.size.z);
            float errorPrimary = Mathf.Abs(actual.x - targetSize.length) + Mathf.Abs(actual.y - targetSize.width);
            float errorSwap = Mathf.Abs(actual.x - targetSize.width) + Mathf.Abs(actual.y - targetSize.length);
            float candidateError = Mathf.Min(errorPrimary, errorSwap);
            
            if (candidateError < bestError)
            {
                bestError = candidateError;
                bestYaw = yaw;
            }
        }
        
        obj.transform.rotation = originalRotation;
        return bestYaw;
    }
    
    bool TryGetFacingYaw(string facingId, out float yaw)
    {
        yaw = 0f;
        if (string.IsNullOrEmpty(facingId))
            return false;
        
        switch (facingId.ToLowerInvariant())
        {
            case "north_wall":
                yaw = 0f;
                return true;
            case "east_wall":
                yaw = 90f;
                return true;
            case "south_wall":
                yaw = 180f;
                return true;
            case "west_wall":
                yaw = 270f;
                return true;
            case "middle of the room":
            case "ceiling":
                yaw = 0f;
                return true;
            default:
                return false;
        }
    }
    
    float NormalizeAngle(float angle)
    {
        return Mathf.Repeat(angle, 360f);
    }
    
    
    
    Bounds GetObjectBounds(GameObject obj)
    {
        Bounds bounds = new Bounds();
        bool hasBounds = false;
        
        Renderer[] renderers = obj.GetComponentsInChildren<Renderer>();
        foreach (Renderer renderer in renderers)
        {
            Bounds childBounds = renderer.localBounds;
            Transform childTransform = renderer.transform;
            
            Vector3 center = childBounds.center;
            Vector3 extents = childBounds.extents;
            
            for (int xSign = -1; xSign <= 1; xSign += 2)
            {
                for (int ySign = -1; ySign <= 1; ySign += 2)
                {
                    for (int zSign = -1; zSign <= 1; zSign += 2)
                    {
                        Vector3 localCorner = center + Vector3.Scale(extents, new Vector3(xSign, ySign, zSign));
                        Vector3 worldCorner = childTransform.TransformPoint(localCorner);
                        
                        if (!hasBounds)
                        {
                            bounds = new Bounds(worldCorner, Vector3.zero);
                            hasBounds = true;
                        }
                        else
                        {
                            bounds.Encapsulate(worldCorner);
                        }
                    }
                }
            }
        }
        
        if (!hasBounds)
        {
            bounds = new Bounds(obj.transform.position, Vector3.zero);
        }
        
        return bounds;
    }
    
    [ContextMenu("Reload Scene")]
    public void ReloadScene()
    {
        // Clear existing objects
        GameObject[] existingObjects = GameObject.FindGameObjectsWithTag("Untagged");
        foreach (GameObject obj in existingObjects)
        {
            if (obj != this.gameObject)
            {
                DestroyImmediate(obj);
            }
        }
        
        // Reload
        Start();
    }
}
