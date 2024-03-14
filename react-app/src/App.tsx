import Alert from "./components/Alert";
import Button from "./components/Button";
import { useState } from "react";

function App() {
	const [alertVisible, setAlertVisibility] = useState(false);

	return (
		<div> 
			{ alertVisible && <Alert onClose={() => setAlertVisibility(!alertVisible)}>你好世界</Alert>}
			<Button color="secondary" onClick={() => setAlertVisibility(!alertVisible)}>X</Button>
		</div>
	);
	// let items = [
	// 	"New York",
	// 	"Hong Kong", 
	// 	"London",
	// 	"Pennsylvania"
	// ];	

	// const handleSelectedItem = (item: string) => {
	// 	console.log(item);
	// }

	// return <div><ListGroup items={items} heading={"Cities"} selectItem={handleSelectedItem}/></div> 
}

export default App;
