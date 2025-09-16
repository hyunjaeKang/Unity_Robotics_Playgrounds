using UnityEngine;
using Unity.MLAgents;


public class DroneSetting : MonoBehaviour
{
    public GameObject DroneAgent;
    public GameObject Goal;

    UnityEngine.Vector3 areaInitPos;
    UnityEngine.Vector3 droneInitPos;
    UnityEngine.Quaternion droneInitRot;

    EnvironmentParameters m_ResetParams;

    private Transform AreaTrans;
    private Transform DroneTrans;
    private Transform GoalTrans;

    private Rigidbody DroneAgent_Rigidbody;

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        Debug.Log(m_ResetParams);

        AreaTrans = gameObject.transform;
        DroneTrans = DroneAgent.transform;
        GoalTrans = Goal.transform;

        areaInitPos = AreaTrans.position;
        droneInitPos = DroneTrans.position;
        droneInitRot = DroneTrans.rotation;

        DroneAgent_Rigidbody = DroneAgent.GetComponent<Rigidbody>();
    }

    // Update is called once per frame
    public void AreaSetting()
    {
        DroneAgent_Rigidbody.linearVelocity = Vector3.zero;
        DroneAgent_Rigidbody.angularVelocity = Vector3.zero;

        DroneTrans.position = droneInitPos;
        DroneTrans.rotation = droneInitRot;

        GoalTrans.position = areaInitPos + new Vector3(Random.Range(-5f, 5f), Random.Range(-5f, 5f), Random.Range(-5f, 5f));
    }
}
