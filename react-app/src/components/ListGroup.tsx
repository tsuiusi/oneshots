// import { Fragment } from "react";
// Can be achieved by just using <> and not importing fragment
// import { MouseEvent } from "react";
import {useState} from "react";


interface Props {
	items: string[];
	heading: string;
}

// a component cannot return more than 1 element (listgroup can only return listgroup and not h1)
// this can be solved by wrapping the whole thing in a div or fragments
function ListGroup({ items, heading }: Props) {
	// Hook: tells React this component can have data or states that change over time
	const [selectedIndex, setSelectedIndex] = useState(-1);

	// const getMessage = () => {
	// 	return items.length === 0 ? <p>empty</p> : null;
	// }

	// Event handler
	// const handleClick = (event: MouseEvent) => {
	// 	console.log(event)
	// }

	return (
		<>	
			<h1>{heading}</h1>
			{items.length === 0 && <p>empty</p>}
			<ul className="list-group">
				{
					// If we're retrieving items from an API it's usually a class
					// So we'll list a property (e.g item.id) instead of just item
					items.map((item, index) => (<li 
						className={selectedIndex === index ? 'list-group-item active' : 'list-group-item'}
						key={item} 
						onClick={() => { setSelectedIndex(index); }}
					>
						{item}
					</li>))
				}
			</ul>
		</>
	);
}


export default ListGroup;