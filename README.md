# Russel

## Inspiration  
Growing up, many of us have witnessed firsthand how difficult life can be for disabled individuals, the elderly, and those who are chronically ill. Simple tasks that most of us take for granted—like walking to get a glass of water or taking medicine—become monumental challenges. These moments can be isolating, frustrating, and, at times, dangerous. Two of our team members have mothers who are struggling with health conditions, which made this project personal for us. We want to make a difference in the lives of people who face these challenges every day, and we believe technology can help.

## What it does  
Our autonomous robot, Russel - inspired by the little boy from UP, is designed to bring independence back to those who need it most. It can receive voice commands from the user, locate an item, move to it, pick it up, and bring it back to the user. Our robot is also equipped with the ability to climb stairs, making it a versatile helper in various environments.

## How we built it  
We focused on four core features to bring this idea to life:

1. **Autonomous Movement**: Our robot moves freely around the environment, navigating obstacles and carrying out tasks.
2. **Speech to Action**: The user can speak to the robot to issue commands, making it easy to interact.
3. **Picking Items Using Arms**: The robot has a robotic arm capable of picking up objects, from water bottles to medications.
4. **Navigating with Whegs**: We’ve integrated whegs, a combination of wheels and legs, that enable the robot to navigate over uneven terrain and even climb stairs.

At the heart of our robot is the **NVIDIA Jetson Orin Nano**, and we are extremely grateful to the Nvidia team for providing us with this powerful tool and supporting us throughout the hackathon.

We used an **Arduino** to control the motors for the wheels, which is connected to the Jetson for coordination. Other components include:

- Brushless motors for efficient movement
- IntelliSense depth cameras for object detection
- A webcam for visual input
- A robotic arm for grasping items
- A laser-cut chassis for structural integrity
- ROS2 for robot control
- SAM + VLM for enhanced item detection and segmentation
- AutoCAD for 3D printing parts
- Soldered circuits to bring everything together

## Challenges we ran into  
One of our biggest challenges was designing and building the circuits from scratch. We didn’t have the right connection wire for the Arduino to the Jetson, so we had to figure out how to wire it up and troubleshoot issues along the way. Each team member stepped up in different ways, learning new skills as we went. We also had to find creative solutions to integrate all the hardware, software, and communication systems into one cohesive unit.

## Accomplishments that we're proud of  
In just 36 hours, we built a fully functional, product-grade autonomous robot from the ground up. The fact that we were able to incorporate so many advanced features into such a short timeframe is something we are truly proud of. But beyond the technical achievement, knowing that our work could make a meaningful difference in someone’s life makes it all worth it.

## What we learned  
This project pushed all of us out of our comfort zones. As a team, we had to learn how to design and build robot parts, write code for a complex system, and put together intricate circuitry—all while staying focused on our ultimate goal of improving lives. Our experience taught us the power of teamwork and how valuable it is to keep learning. We also learned how to improvise and innovate when faced with unexpected challenges.

## What's next for Russel
Looking ahead, we plan to expand the robot’s capabilities. We want to integrate more IoT devices, such as smart home systems, to further enhance its utility. Our ultimate goal is to bring this technology into production, helping people with disabilities and elderly individuals lead more independent and fulfilling lives. This project is just the beginning, and we are excited about the potential impact it can have.
